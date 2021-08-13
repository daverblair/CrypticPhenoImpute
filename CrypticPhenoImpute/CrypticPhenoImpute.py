import argparse
from vlpi.vLPI import vLPI
from vlpi.data.ClinicalDataset import ClinicalDataset,ClinicalDatasetSampler
from vlpi.data.ICDUtilities import ICD_PATH,ICD10TranslationMap,ICDUtilities
import numpy as np
import pandas as pd
import sys
import pickle
import os
import re
import pkg_resources
import wget

DATA_PATH = pkg_resources.resource_filename('CrypticPhenoImpute', 'Data/')
icdClass=ICDUtilities()
__version__ = "0.1.5"

def main():

    #fixed data loaded into memory
    dis_table = pd.read_csv(DATA_PATH+"TargetDiseaseCodes.txt",sep='\t',index_col="CODE")

    parser = argparse.ArgumentParser(description='Imputes the cryptic phenotypes analyzed in Blair et al. 2021 into arbitrary clinical datasets.')

    parser.add_argument("encoding",help="ICD encoding. Must be either 'ICD10-CM' or 'ICD10-UKBB'.",type=str)

    parser.add_argument("datafile",help="Path to the datafile containing the clinical information. Note, the software expects a tab-delimitted text file with two columns. The first column contains a unique ID for every subject. The second column contains a comma-separated list of diagnosed ICD10 codes. DO NOT include a header.",type=str)

    parser.add_argument("cryptic_phenotype",help="Disease cryptic phenotype to be imputed. Must be in the following list: {0:s}. To see a key for the cryptic phenotypes, provide the argument KEY instead.".format(', '.join(list(dis_table.index))),type=str)

    parser.add_argument("output_file",help="Path to the output file.",type=str)
    parser.add_argument("--convertToUKBB",help="Flag that indicates ICD10-CM codes will be converted to UKBB. Note, will raise error if encoding is already 'ICD10-UKBB'",action="store_true")
    parser.add_argument("--model_path",help="Path to use for Cryptic Phenotype Models. Defaults to a path within the package, which might have write restrictions. If so, use this argument to specify a local path.",type=str)



    args = parser.parse_args()

    if args.cryptic_phenotype=='KEY':
        print(dis_table)
        sys.exit()

    assert args.encoding in ['ICD10-CM','ICD10-UKBB'],"Encoding not recognized. Please use 'ICD10-CM' or 'ICD10-UKBB'."
    assert args.cryptic_phenotype in dis_table.index, "Disease cryptic phenotype to be imputed. Must be in the following list: {0:s}. To see a key for the cryptic phenotypes, provide the argument KEY instead.".format(', '.join(list(dis_table.index)))
    disease_code=dis_table.loc[args.cryptic_phenotype]['OMIM_HPO_ID']

    #initialize the ClinicalDataset class
    if args.encoding=='ICD10-CM':
        currentClinicalDataset=ClinicalDataset()
    else:
        currentClinicalDataset=ClinicalDataset(ICDFilePaths=[ICD_PATH+'icd10_ukbb.txt',ICD_PATH+'ICD10_Chapters.txt'])



    #read the dataset into memory
    currentClinicalDataset.ReadDatasetFromFile(args.datafile,1,indexColumn=0, hasHeader=False,chunkSize = 50000)



    if args.encoding=='ICD10-CM':
        with open(DATA_PATH+'Allowed_ICD10CM.txt') as f:
            icdSetToInclude=f.read().strip().split('\n')
    else:
        with open(DATA_PATH+'Allowed_ICD10UKBB.txt') as f:
            icdSetToInclude=f.read().strip().split('\n')

    currentClinicalDataset.IncludeOnly(icdSetToInclude)

    #tranlated into ICD10-UKBB if desired
    if args.convertToUKBB and args.encoding=='ICD10-CM':
        icd10_ukbb_translation = ICD10TranslationMap()
        icd_ukbb_map={}
        for code in currentClinicalDataset.dxCodeToDataIndexMap.keys():
            icd_ukbb_map[code]=icd10_ukbb_translation.ReturnConversionSet(code)
        currentClinicalDataset.ConstructNewDataArray(icd_ukbb_map)
        args.encoding='ICD10-UKBB'
    elif args.convertToUKBB and args.encoding=='ICD10-UKBB':
        raise ValueError("ICD10 data is reported to already use UKBB encoding. Please double check arguments.")

    #set up the model directories if they do not already exist
    if args.model_path is not None:
        MODEL_PATH=args.model_path
        if MODEL_PATH[-1]!='/':
            MODEL_PATH+='/'
    else:
        MODEL_PATH = pkg_resources.resource_filename('CrypticPhenoImpute', 'Models/')

    try:
        os.mkdir(MODEL_PATH)
    except FileExistsError:
        pass

    try:
        os.mkdir(MODEL_PATH+'ICD10UKBB_Models')
    except FileExistsError:
        pass

    try:
        os.mkdir(MODEL_PATH+'ICD10CM_Models')
    except FileExistsError:
        pass


    hpo_table = pd.read_csv(DATA_PATH+"HPOTable.txt",sep='\t',index_col="HPO_ICD10_ID")
    model_table = pd.read_csv(DATA_PATH+"ModelTable.txt",sep='\t',index_col="OMIM_ICD_ID")
    if args.encoding=='ICD10-CM':
        #load the HPO term table
        disease_hpo=model_table.loc[disease_code]['Annotated HPO Terms UCSF'].split(';')
        hpo_icd10_map = {hpo: hpo_table.loc[hpo]['ICD10'].split(';') for hpo in disease_hpo}

        icd10_HPO_map={}
        for key,value in hpo_icd10_map.items():
            for icd in value:
                try:
                    icd10_HPO_map[icd]+=[key]
                except KeyError:
                    icd10_HPO_map[icd]=[key]

        currentClinicalDataset.ConstructNewDataArray(icd10_HPO_map)
        sampler=ClinicalDatasetSampler(currentClinicalDataset,0.5)
        vlpi_model=vLPI(sampler,model_table.loc[disease_code]['UCSF Max. Model Rank'])

        try:
            vlpi_model.LoadModel(MODEL_PATH+'ICD10CM_Models/{0:s}.pth'.format(disease_code.replace(':','_')))
        except FileNotFoundError:
            print("\nDownloading model files from GitHub.")
            wget.download("https://raw.githubusercontent.com/daverblair/CrypticPhenoImpute/master/CrypticPhenoImpute/Models/ICD10CM_Models/{0:s}.pth".format(disease_code.replace(':','_')),out=MODEL_PATH+'ICD10CM_Models/')
            vlpi_model.LoadModel(MODEL_PATH+'ICD10CM_Models/{0:s}.pth'.format(disease_code.replace(':','_')))


        try:
            model_hpo_index=pd.read_pickle(MODEL_PATH+'ICD10CM_Models/{0:s}_Index.pth'.format(disease_code.replace(':','_')))
        except FileNotFoundError:
            print("\nDownloading index files from GitHub.")
            wget.download("https://raw.githubusercontent.com/daverblair/CrypticPhenoImpute/master/CrypticPhenoImpute/Models/ICD10CM_Models/{0:s}_Index.pth".format(disease_code.replace(':','_')),out=MODEL_PATH+'ICD10CM_Models/')
            with open(MODEL_PATH+'ICD10CM_Models/{0:s}_Index.pth'.format(disease_code.replace(':','_')),'rb') as f:
                model_hpo_index=pickle.load(f)

        ######## This code corrects variations in the order in which symptoms are stored that occurred between an earlier and the current version of the ClinicalDataset class
        ######## Clearly, this is less than ideal, but it wasn't worth refitting all of the models for this small change in storage that could be corrected.
        symptom_array=currentClinicalDataset.ReturnSparseDataMatrix()
        new_order=[currentClinicalDataset.dxCodeToDataIndexMap[x] for x in model_hpo_index.index]
        symptom_array=(symptom_array.tocsr()[:,new_order]).tocoo()
        ########
        ########

        cp=vlpi_model.ComputeEmbeddings(dataArrays=(symptom_array,[]))[:,model_table.loc[disease_code]['UCSF Model-Top Component']]
        output_table=pd.DataFrame({'Subject_ID':currentClinicalDataset.data.index,args.cryptic_phenotype:cp})
        output_table.set_index('Subject_ID',inplace=True,drop=True)
        output_table.to_csv(args.output_file,sep='\t')

    # use the ICD10-UKBB encoding
    else:
        #load the HPO term table
        disease_hpo=model_table.loc[disease_code]['Annotated HPO Terms UKBB'].split(';')
        hpo_icd10_map = {hpo: hpo_table.loc[hpo]['ICD10_UKBB'].split(';') for hpo in disease_hpo}

        icd10_HPO_map={}
        for key,value in hpo_icd10_map.items():
            for icd in value:
                try:
                    icd10_HPO_map[icd]+=[key]
                except KeyError:
                    icd10_HPO_map[icd]=[key]

        currentClinicalDataset.ConstructNewDataArray(icd10_HPO_map)

        sampler=ClinicalDatasetSampler(currentClinicalDataset,0.5)
        vlpi_model=vLPI(sampler,model_table.loc[disease_code]['UCSF Max. Model Rank'])
        try:
            vlpi_model.LoadModel(MODEL_PATH+'ICD10UKBB_Models/{0:s}.pth'.format(disease_code.replace(':','_')))
        except FileNotFoundError:
            print("\nDownloading model files from GitHub.")
            wget.download("https://raw.githubusercontent.com/daverblair/CrypticPhenoImpute/master/CrypticPhenoImpute/Models/ICD10UKBB_Models/{0:s}.pth".format(disease_code.replace(':','_')),out=MODEL_PATH+'ICD10UKBB_Models/')
            vlpi_model.LoadModel(MODEL_PATH+'ICD10UKBB_Models/{0:s}.pth'.format(disease_code.replace(':','_')))


        try:
            model_hpo_index=pd.read_pickle(MODEL_PATH+'ICD10UKBB_Models/{0:s}_Index.pth'.format(disease_code.replace(':','_')))
        except FileNotFoundError:
            print("\nDownloading index files from GitHub.")
            wget.download("https://raw.githubusercontent.com/daverblair/CrypticPhenoImpute/master/CrypticPhenoImpute/Models/ICD10UKBB_Models/{0:s}_Index.pth".format(disease_code.replace(':','_')),out=MODEL_PATH+'ICD10UKBB_Models/')
            with open(MODEL_PATH+'ICD10UKBB_Models/{0:s}_Index.pth'.format(disease_code.replace(':','_')),'rb') as f:
                model_hpo_index=pickle.load(f)

        symptom_array=currentClinicalDataset.ReturnSparseDataMatrix()
        new_order=[currentClinicalDataset.dxCodeToDataIndexMap[x] for x in model_hpo_index.index]
        symptom_array=(symptom_array.tocsr()[:,new_order]).tocoo()
        ########
        ########

        cp=vlpi_model.ComputeEmbeddings(dataArrays=(symptom_array,[]))[:,model_table.loc[disease_code]['UKBB Model-Top Component']]
        output_table=pd.DataFrame({'Subject_ID':currentClinicalDataset.data.index,args.cryptic_phenotype:cp})
        output_table.set_index('Subject_ID',inplace=True,drop=True)
        output_table.to_csv(args.output_file,sep='\t')

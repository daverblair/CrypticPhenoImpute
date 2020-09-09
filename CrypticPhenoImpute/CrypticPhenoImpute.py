import argparse
from vlpi.vLPI import vLPI
from vlpi.data.ClinicalDataset import ClinicalDataset,ClinicalDatasetSampler
from vlpi.data.ICDUtilities import ICD_PATH
import sklearn
import numpy as np
import pandas as pd
import sys
import pickle
import os
import re

if __name__=='__main__':

    dis_table = pd.read_csv("../Data/TargetDiseaseCodes.txt",sep='\t',index_col="CODE")

    parser = argparse.ArgumentParser(description='Imputes the cryptic phenotypes analyzed in Blair et al. 2020 into arbitrary clinical datasets.')

    parser.add_argument("encoding",help="ICD encoding. Must be either 'ICD10-CM' or 'ICD10-UKBB'.",type=str)

    parser.add_argument("datafile",help="Path to the datafile containing the clinical information. Note, the software expects a tab-delimitted text file with two columns. The first column contains a unique ID for every subject. The second column contains a comma-separated list of diagnosed ICD10 codes. DO NOT include a header.",type=str)

    parser.add_argument("cryptic_phenotype",help="Disease cryptic phenotype to be imputed. Must be in the following list: {0:s}. To see a key for the cryptic phenotypes, provide the argument KEY instead.".format(', '.join(list(dis_table.index))),type=str)

    parser.add_argument("output_file",help="Path to the output file.",type=str)

    parser.add_argument("--use_best",help="Disease cryptic phenotype to be imputed. Must be in the following list: {0:s}. To see a key for the cryptic phenotypes, provide the argument KEY instead.".format(', '.join(list(dis_table.index))),action="store_true")

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


    #if using ICD10-CM, use the vlpi model directly. Requires translating from ICD10-CM into HPO terms

    if args.encoding=='ICD10-CM':
        #load the HPO term table
        hpo_table = pd.read_csv("../Data/HPOTable.txt",sep='\t',index_col="HPO_ICD10_ID")
        model_table = pd.read_csv("../Data/ModelTable.txt",sep='\t',index_col="Disease ID")

        disease_hpo=model_table.loc[disease_code]['Annotated HPO Terms'].split(',')
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

        vlpi_model=vLPI(sampler,model_table.loc[disease_code]['Max. Model Rank'])
        vlpi_model.LoadModel('../Data/ICD10CM_Models/{0:s}.pth'.format(disease_code.replace(':','_')))

        ######## This code corrects variations in the order in which symptoms are stored that occurred between an earlier and the current version of the ClinicalDataset class
        ######## Clearly, this is less than ideal, but it wasn't worth refitting all of the models for this small change in storage that could be corrected.
        symptom_array=currentClinicalDataset.ReturnSparseDataMatrix()
        with open('../Data/ICD10CM_Models/{0:s}_Index.pth'.format(disease_code.replace(':','_')),'rb') as f:
            model_hpo_index=pickle.load(f)

        new_order=[currentClinicalDataset.dxCodeToDataIndexMap[x] for x in model_hpo_index.keys()]
        symptom_array=(symptom_array.tocsr()[:,new_order]).tocoo()
        ########
        ########

        cp=vlpi_model.ComputeEmbeddings(dataArrays=(symptom_array,[]))[:,model_table.loc[disease_code]['Top Component']]
        output_table=pd.DataFrame({'Subject_ID':currentClinicalDataset.data.index,args.cryptic_phenotype:cp})
        output_table.set_index('Subject_ID',inplace=True,drop=True)
        output_table.to_csv(args.output_file,sep='\t')
    else:
        if args.use_best==True:
            features=pd.read_csv('../Data/ICD10UKBB_Models/{0:s}/TopModelFeatures.txt'.format(disease_code.replace(':','_')),sep='\t',header=None)
            currentClinicalDataset.IncludeOnly(features[0].values)
            symptom_array=currentClinicalDataset.ReturnSparseDataMatrix()
            top_model=os.listdir('../Data/ICD10UKBB_Models/{0:s}/TopModel'.format(disease_code.replace(':','_')))
            r=re.compile('.*.pth')
            top_model=list(filter(r.match,top_model))[0]
            with open('../Data/ICD10UKBB_Models/{0:s}/TopModel/{1:s}'.format(disease_code.replace(':','_'),top_model),'rb') as f:
                model_dict = pickle.load(f)
            cp = model_dict['Model'].predict(symptom_array)

        else:

            cp = np.zeros((currentClinicalDataset.numPatients))
            features=pd.read_csv('../Data/ICD10UKBB_Models/{0:s}/BaggedModelFeatures.txt'.format(disease_code.replace(':','_')),sep='\t')
            currentClinicalDataset.IncludeOnly(features['ICD10'].values)
            symptom_array=currentClinicalDataset.ReturnSparseDataMatrix()


            all_models=os.listdir('../Data/ICD10UKBB_Models/{0:s}/BaggedModels'.format(disease_code.replace(':','_')))
            r=re.compile('.*.pth')
            all_models=list(filter(r.match,all_models))
            num_models=0
            for model_string in all_models:
                num_models+=1
                with open('../Data/ICD10UKBB_Models/{0:s}/BaggedModels/{1:s}'.format(disease_code.replace(':','_'),model_string),'rb') as f:
                    model_dict = pickle.load(f)
                cp += model_dict['Model'].predict(symptom_array)
            cp/=num_models

        output_table=pd.DataFrame({'Subject_ID':currentClinicalDataset.data.index,args.cryptic_phenotype:cp})
        output_table.set_index('Subject_ID',inplace=True,drop=True)
        output_table.to_csv(args.output_file,sep='\t')

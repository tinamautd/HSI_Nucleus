import os               # To use operating system dependent functionality
import tensorflow as tf

import scipy.io as io   
from sklearn import metrics
import h5py             
import matplotlib.pyplot as plt
import time            
import numpy as np      
import pandas as pd 

from CNN import cnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To mute tf warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "7" # Choose GPU to use

# change current path to TARGET
TARGET_DIR = '/HSI_Nucleus'
os.chdir(TARGET_DIR)

DATA_DIR = '/HSI_Nucleus/Data'

VALI_FOLD =  [154,161]
sheet_name = 'Val_154-161'
TEST_FOLD = [103,120,127,133]

threshold = 0.5

CKPT_NAME =  'CkPt-22-0.87'

NUM_CLASSES = 2

learning_rate = 1e-5
decay=0
dropout = 0.3
BATCH_SIZE = 16

RESULT_DIR = 'Results/T'+str(TEST_FOLD)+'V'+str(VALI_FOLD)+'LR'+str(learning_rate)+'_DR'+str(dropout)+'_DC'+str(decay)+'_BS'+str(BATCH_SIZE)


model = cnn(dropout=DR)
model.load_weights(os.path.join(os.getcwd(),RESULT_DIR,'Checkpoints',CKPT_NAME+'.hdf5'),'r')

# Compile the model
myOptimizer = tf.keras.optimizers.Adam(lr=learning_rate)
model.compile(loss='binary_crossentropy',optimizer=myOptimizer,metrics=['accuracy'])

"""load validation data"""
print('*******Load validation data*******')
startTimePoint = time.time()
FirstValidNum=True
i_th = 0

list_val = []

for P_NUM in TEST_FOLD:

	folder = DATA_DIR + '/P_' + str(P_NUM)
	i_th += 1
	print('The '+str(i_th)+'_th PT is P_' + str(P_NUM))
	count = 0; # count of nuclei
	FirstNum = True # first patch for each PT
	FirstPtInd = True

	list = [filename for filename in os.listdir(folder) if filename.endswith('.mat')]
	list_val = list_val + list 


	
	for filename in list:

		f = h5py.File(os.path.join(folder,filename),'r')

		data = np.array(f['hsi'])
		label = np.array(f['label'],dtype=np.int8)

		NUM_DIMS = np.ndim(data)# Check data dimension
		if NUM_DIMS == 3:
			data = np.expand_dims(data,axis=0)
		data = np.transpose(data,(0,3,2,1)) # for v7.3 .mats, otherwise 3,0,1,2
		count = count + 1

		if(FirstNum==True):
			temp_data = data
			temp_label = label
			FirstNum=False
		else:
			temp_data = np.concatenate((temp_data,data),axis=0)
			temp_label= np.concatenate((temp_label,label),axis=0)

		if (np.mod(count,100)==0) or (count==len(list)):
			if(FirstPtInd==True):
				pt_data = temp_data
				pt_label = temp_label
				FirstPtInd = False
				FirstNum = True
				#print(np.shape(pt_data))
			else:
				pt_data = np.concatenate((pt_data,temp_data),axis=0)
				pt_label= np.concatenate((pt_label,temp_label),axis=0)
				FirstNum = True

	if(FirstValidNum==True):
		valiData = pt_data
		valiLabel= pt_label
		FirstValidNum=False
	else:
		valiData = np.concatenate((valiData,pt_data),axis=0)
		valiLabel= np.concatenate((valiLabel,pt_label),axis=0)
	# Next patient
	

	print('Evaluate on valiData')
	results = model.evaluate(valiData,valiLabel)
	print('vali loss, vali acc:')
	print(results)
	startTimePoint = time.time()
	predictions = model.predict(valiData)
	endTimePoint = time.time()
	print(np.shape(valiData))
	print(str(endTimePoint-startTimePoint)+' seconds')
	
	fpr, tpr, thresholds = metrics.roc_curve(valiLabel, predictions)
	auc = metrics.auc(fpr, tpr)
	print('AUC: '+str(auc))
	optimal_idx = np.argmax(tpr - fpr)
	optimal_threshold = thresholds[optimal_idx]
	print('opt_th = '+str(optimal_threshold))

	predicted_labels = predictions>threshold
	acc = metrics.accuracy_score(valiLabel,predicted_labels)
	print('Accuracy: '+str(acc))
	sens = metrics.recall_score(valiLabel, predicted_labels, pos_label=1)
	print('Sensitivity: '+str(sens))
	spec = metrics.recall_score(valiLabel, predicted_labels, pos_label=0)
	print('Specificity: '+str(spec))
	confusion = metrics.confusion_matrix(valiLabel, predicted_labels)
	print('confusion matrix:')
	print(confusion)

	predicted_labels_opt = predictions>optimal_threshold
	acc_opt = metrics.accuracy_score(valiLabel,predicted_labels_opt)
	print('Accuracy (Th = .opt): '+str(acc_opt))
	sens_opt = metrics.recall_score(valiLabel, predicted_labels_opt, pos_label=1)
	print('Sensitivity (Th = .opt): '+str(sens_opt))
	spec_opt = metrics.recall_score(valiLabel, predicted_labels_opt, pos_label=0)
	print('Specificity (Th = opt): '+str(spec_opt))
	confusion_opt = metrics.confusion_matrix(valiLabel, predicted_labels_opt)
	print('Confusion (Th = opt):')
	print(confusion_opt)

	## Save all evaluation results
	#io.savemat(os.path.join(RESULT_DIR,('Evaluation-'+CKPT_NAME+'_VAL'+str(VALI_FOLD)+'.mat')),#+'_Phase'+str(learning_phase)+'.mat')),
	io.savemat(os.path.join('Results',('Testing-'+str(P_NUM)+'_'+CKPT_NAME+'_VAL'+str(VALI_FOLD)+'.mat')),
		{'valiLabels':valiLabel,
		'prediction':predictions,
		'predicted_labels':predicted_labels,
		'predicted_labels_opt':predicted_labels_opt,
		'auc':auc,'fpr':fpr,'tpr':tpr,
		'optimal_threshold':optimal_threshold,
		'acc':acc,'sens':sens,'spec':spec,
		'acc_opt':acc_opt,'sens_opt':sens_opt,'spec_opt':spec_opt
		})
	print('Saved!')

	## Plot ROC
	plt.figure(figsize=(8,8))
	plt.plot(fpr,tpr)
	plt.title('AUC = '+str(auc))
	plt.ylabel('TPR')
	plt.xlabel('FPR')
	#plt.savefig(os.path.join(RESULT_DIR,'ROC (AUC='+str(auc)+')_'+CKPT_NAME+'_VAL'+str(VALI_FOLD)+'.png'))#+'_Phase'+str(learning_phase)+'.png'))
	plt.savefig(os.path.join('Results','Testing-'+str(P_NUM)+'_'+'ROC (AUC='+str(auc)+')_'+CKPT_NAME+'_TEST'+str(TEST_FOLD)+'.png'))
	#plt.show()

	#-------------------------------------------------------------
	#print('Filename dimension '+ str(len(list_val)))
	new_list = {'Filename':list_val,
				'Label':np.squeeze(valiLabel),
				'Predictions':np.squeeze(predictions)}
	excel_name = 'Results/'+'Testing-'+str(P_NUM)+'_'+CKPT_NAME+'.xlsx'
	#excel_name = RESULT_DIR + '/Results.xlsx'
	df = pd.DataFrame(data = new_list,index=np.arange(0,len(list_val),1))
	writer = pd.ExcelWriter(excel_name,engine = 'xlsxwriter')
	df.to_excel(writer,sheet_name = sheet_name,index = True)
	writer.save()


'''
print(np.shape(valiLabel))
print(np.shape(predicted_labels))
print(np.shape(predictions))

new_list = {'Label':np.squeeze(valiLabel)}
excel_name = RESULT_DIR + '/' + 'Results1.xlsx'
df = pd.DataFrame(data = new_list,index=np.arange(0,len(list_val),1))
writer = pd.ExcelWriter(excel_name,engine = 'xlsxwriter')
df.to_excel(writer,sheet_name = sheet_name,index = True)
writer.save()


new_list = {'pred_label':np.squeeze(predicted_labels)}
excel_name = RESULT_DIR + '/' + 'Results2.xlsx'
df = pd.DataFrame(data = new_list,index=np.arange(0,len(list_val),1))
writer = pd.ExcelWriter(excel_name,engine = 'xlsxwriter')
df.to_excel(writer,sheet_name = sheet_name,index = True)
writer.save()


new_list = {'Predictions':np.squeeze(predictions)}
excel_name = RESULT_DIR + '/' + 'Results3.xlsx'
df = pd.DataFrame(data = new_list,index=np.arange(0,len(list_val),1))
writer = pd.ExcelWriter(excel_name,engine = 'xlsxwriter')
df.to_excel(writer,sheet_name = sheet_name,index = True)
writer.save()
'''
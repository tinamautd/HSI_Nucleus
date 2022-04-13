import os
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import scipy.io as io   
import h5py            
import sys              
import matplotlib.pyplot as plt
import time             
import sklearn.metrics as metrics 
import shutil
import random

#--------------------------------------------------
""" INITIALIZATION """
#--------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To mute tf warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1" # Choose GPU to use

# change current path to TARGET 
TARGET_DIR = '/HSI_Nucleus'
os.chdir(TARGET_DIR) 

DATA_DIR = 'Data'

#--------------------------------------------------
""" HYPER-PARAMETERS """
#--------------------------------------------------
NUM_CLASSES = 2
BATCH_SIZE = 16  
num_epochs = 30
learning_rate = 1e-4 # 1e-5 for some folds
dropout = 0.1 # 0.2 or 0.3 for some folds
l2_value=0 

PT_INDEX = [62,68,74,103,110,120,127,133,134,137,146,149,154,161,166,172,174,184,187,188] 
VALI_FOLDs = [[68,133],[62,74],[134,137],[146,149],[154,161],[166,172],[174,184],[187,188]]
TEST_FOLD= [103,110,120,127]

print('****************************************************************')

#--------------------------------------------------
""" LOAD ALL DATA INTO MEMORY FOR FAST TRAINING """
#--------------------------------------------------
for valid_fold in range(0, np.size(VALI_FOLDs,0)): # for CV

	VALI_FOLD = VALI_FOLDs[valid_fold]
	print('VALID FOLD is: '+ str(VALI_FOLD))
	print('TEST FOLD is: ' + str(TEST_FOLD))
	print('******** LOADING DATA *********')

	startTimePoint = time.time()

	FirstTrainNum=True
	FirstValidNum=True
	i_th = 0

	# FOR EACH PATIENT
	for P_NUM in PT_INDEX:	
		
		if P_NUM in TEST_FOLD:
			continue

		folder = DATA_DIR + '/P_' + str(P_NUM)
		i_th += 1
		print('The '+str(i_th)+'_th PT is P_' + str(P_NUM))
		count = 0; # count of nuclei
		FirstNum = True # first patch for each PT
		FirstPtInd = True


		# ALL NUCLEI .MAT FILES IN THE PT FOLDER
		list = [filename for filename in os.listdir(folder) if filename.endswith('.mat')]
		
		for filename in list:
			
			f = h5py.File(os.path.join(folder,filename),'r')
			data = np.array(f['patches_hsi']) # augmented patches, for RGB, use 'patches_rgb'
			label = np.array(f['label'],dtype=np.int8)
			label = np.concatenate((label,label,label,label),axis=0)
			label = np.squeeze(label)

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
				else:
					pt_data = np.concatenate((pt_data,temp_data),axis=0)
					pt_label= np.concatenate((pt_label,temp_label),axis=0)
					FirstNum = True



		# ONCE ALL NUCLEI OF ONE PT ARE LOADED, SORT THEM TO TRAIN/VALID DATA
		print('The data size of PT #'+str(P_NUM)+' is:')
		print(np.shape(pt_data))
		print(np.shape(pt_label))

		if not ((P_NUM in VALI_FOLD) or (P_NUM in TEST_FOLD)):
			if(FirstTrainNum==True):
				trainingData = pt_data
				trainingLabel= pt_label
				FirstTrainNum=False
			else:
				trainingData = np.concatenate((trainingData, pt_data), axis=0)
				trainingLabel= np.concatenate((trainingLabel,pt_label),axis=0)

		elif (P_NUM in VALI_FOLD):
			if(FirstValidNum==True):
				valiData = pt_data
				valiLabel= pt_label
				FirstValidNum=False
			else:
				valiData = np.concatenate((valiData,pt_data),axis=0)
				valiLabel= np.concatenate((valiLabel,pt_label),axis=0)
		# Next patient

	endTimePoint = time.time()	
	loadingTime = (endTimePoint-startTimePoint)/60
	print('Done loading data')
	print('Loading time = '+str(loadingTime)+' minutes')
	
	print('Training data: ')
	print(np.shape(trainingData))
	print('Training label: ')
	print(np.shape(trainingLabel))
	print('Validation data: ')
	print(np.shape(valiData))
	print('Validation label: ')
	print(np.shape(valiLabel))

	"""import CNN network"""
	sys.path.append("Architectures/")
	from CNN import * 
			
	RESULT_DIR = 'Results/'+'T'+str(TEST_FOLD)+'V'+str(VALI_FOLD)+'LR'+str(learning_rate)+'_DR'+str(dropout)+'_BS'+str(BATCH_SIZE)
	if not os.path.isdir(RESULT_DIR):
		os.makedirs(RESULT_DIR)
		os.makedirs(RESULT_DIR+'/Checkpoints')
		print('*******Result DIR created*******')

	model = cnn(input_shape=(101,101,87),dropout=dropout,regularizer=tf.keras.regularizers.l2(l2_value))
	# model = cnn(input_shape=(101,101,3),dropout=dropout,regularizer=tf.keras.regularizers.l2(l2_value))

	# Compile the model
	myOptimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1=0.9,beta_2=0.999) 
	model.compile(loss='binary_crossentropy',optimizer=myOptimizer,metrics=['accuracy'])
	
	# callback to save checkpoint
	filepath = os.path.join(RESULT_DIR+'/Checkpoints/CkPt-{epoch:02d}-{val_accuracy:.2f}.hdf5')
	callbacks = [
	    tf.keras.callbacks.EarlyStopping(monitor='loss',patience=3, min_delta=0, verbose=1),
	    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=10, min_delta=0, verbose=1),
	    # tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2, min_lr=1e-8, verbose=1),
	    tf.keras.callbacks.ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,save_best_only=True, mode='max')
	]
	
	# start training
	startTrainingTime = time.time()

	print('******* Fit model with training data *******')
	history = model.fit(trainingData,
						trainingLabel,
						epochs = num_epochs,
						batch_size = BATCH_SIZE,
						shuffle = True,
						verbose=2, # 0=silent, 1=progress bar, 2=one line per epoch
						validation_data=(valiData,valiLabel),
						callbacks=callbacks)

	endTrainingTime = time.time()
	trainingTime = (endTrainingTime-startTrainingTime)/60 # minutes
	print('*******Training is DONE*******')
	print('Training time is '+str(trainingTime))


	'''EVALUATION'''
	# Display all data in training history
	print('History keys are:')
	print(history.history.keys())
	acc = history.history['accuracy']
	loss =history.history['loss'] 

	plt.figure(figsize=(8,4))
	# summarize history for accuracy
	plt.subplot(121)
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	# summarize history for loss
	plt.subplot(122)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper right')
	plt.savefig(os.path.join(RESULT_DIR,'Evaluation.png'))
	#plt.show()
	#model.save(os.path.join(RESULT_DIR,'model.hdf5')) 
#####################################################################################################
#####################################################################################################
	print('Evaluate on valiData')
	results = model.evaluate(valiData,valiLabel)
	print('vali loss, vali acc:')
	print(results)
	predictions = model.predict(valiData)

	fpr, tpr, thresholds = metrics.roc_curve(valiLabel, predictions)
	auc = metrics.auc(fpr, tpr)
	print('AUC: '+str(auc))
	optimal_idx = np.argmax(tpr - fpr)
	optimal_threshold = thresholds[optimal_idx]
	print('opt_th = '+str(optimal_threshold))

	predicted_labels = predictions>0.5
	acc = metrics.accuracy_score(valiLabel,predicted_labels)
	print('Accuracy: '+str(acc))
	sens = metrics.recall_score(valiLabel, predicted_labels, pos_label=1)
	print('Sensitivity: '+str(sens))
	spec = metrics.recall_score(valiLabel, predicted_labels, pos_label=0)
	print('Specificity: '+str(spec))
	confusion = metrics.confusion_matrix(valiLabel, predicted_labels)

	predicted_labels_opt = predictions>optimal_threshold 
	acc_opt = metrics.accuracy_score(valiLabel,predicted_labels_opt)
	print('Accuracy (Th = .opt): '+str(acc_opt))
	sens_opt = metrics.recall_score(valiLabel, predicted_labels_opt, pos_label=1)
	print('Sensitivity (Th = .opt): '+str(sens_opt))
	spec_opt = metrics.recall_score(valiLabel, predicted_labels_opt, pos_label=0)
	print('Specificity (Th = opt): '+str(spec_opt))
	confusion_opt = metrics.confusion_matrix(valiLabel, predicted_labels_opt)


	## Save all evaluation results
	io.savemat(os.path.join(RESULT_DIR,('Evaluation_Final_Model.mat')),#+'_Phase'+str(learning_phase)+'.mat')),
		{'valiLabels':valiLabel,
		'prediction':predictions, 
		'predicted_labels':predicted_labels,
		'predicted_labels_opt':predicted_labels_opt,
		'auc':auc,'fpr':fpr, 'tpr':tpr,
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
	plt.savefig(os.path.join(RESULT_DIR,'ROC (AUC='+str(auc)+')_Final_Model.png'))#+'_Phase'+str(learning_phase)+'.png'))
	#plt.show()

	del model
	tf.keras.backend.clear_session()
"""
Just global settings which to use. This will also potentially have to 
have functions with which to change the values of some of the 
variables here.
"""
act_li = ['climbingdown', 'climbingup', 'jumping', 'lying', 'running',
	'sitting', 'standing', 'walking']
sen_li = ['acc', 'gps', 'gyr', 'lig', 'mag', 'mic']
pos_li = ['chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist']
act_map = dict(zip(act_li, range(1,len(act_li)+1,1)))
sen_map = dict(zip(sen_li, range(1,len(sen_li)+1,1)))
pos_map = dict(zip(pos_li, range(1,len(pos_li)+1,1)))
data_path = '/mnt/6EE804CEE804970D/Academics/Projects/589-Project/Datasets/realworld2016_dataset'
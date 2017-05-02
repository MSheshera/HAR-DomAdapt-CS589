"""
Just global settings which to use. This will also potentially have to 
have functions with which to change the values of some of the 
variables here.
"""
act_li = ['climbingdown', 'climbingup', 'jumping', 'lying', 'running',
	'sitting', 'standing', 'walking']
sen_li = ['acc', 'gps', 'gyr', 'lig', 'mag', 'mic']
pos_li = ['chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist']
usr_li = ['1','2','3','5','8','9','10','11','12','13','15']
# These groups were roughly based on how different or similar the subjects 
# were.
groups = [['2','9','10','15'], ['13','10','9','8'], ['10','2','3','9'], 
	['11','8','15','5']]
tl_groups_diff = [
	{'source':['2','9','10'],'target':['15']},
	{'source':['13','10','9'],'target':['8']},
	{'source':['8','15','11'],'target':['2']},
	{'source':['5','11','8'],'target':['2']}
	]
tl_groups_similar = [
	{'source':['10','2','3'],'target':['9']},
	{'source':['11','8','15'],'target':['5']},
	{'source':['13','1','10'],'target':['3']},
	{'source':['13','10','3'],'target':['12']}
	]
random_seeds = [34, 28, 46, 3, 0]

act_map = dict(zip(act_li, range(1,len(act_li)+1,1)))
sen_map = dict(zip(sen_li, range(1,len(sen_li)+1,1)))
pos_map = dict(zip(pos_li, range(1,len(pos_li)+1,1)))

data_path = '/mnt/6EE804CEE804970D/Academics/Projects/589-Project/Datasets/realworld2016_dataset'
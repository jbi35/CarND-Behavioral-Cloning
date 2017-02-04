import pickle
path_to_log_data="own_data/"
driving_log = open(path_to_log_data + "driving_log.csv","r").readlines()

new_driving_log = []

for l in driving_log:
    center_img,left_img,right_img,steering,throttle,brake,speed = l.split(", ")
    center_img = path_to_log_data + center_img #'/'.join(center_img.split('/')[-3:])
    left_img   = path_to_log_data + left_img   #'/'.join(left_img.split('/')[-3:])
    right_img  = path_to_log_data + right_img  #'/'.join(right_img.split('/')[-3:])
    new_driving_log.append((center_img,float(steering),float(throttle),float(brake),float(speed)))
    new_driving_log.append((left_img,float(steering),float(throttle),float(brake),float(speed)))
    new_driving_log.append((right_img,float(steering),float(throttle),float(brake),float(speed)))

with open("own_data/driving_log.pickle", "wb") as f:
    pickle.dump(new_driving_log,f)

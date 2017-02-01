import pickle
driving_log = open("udacity_data/my_driving_log.csv", "r").readlines()

new_driving_log = []

for l in driving_log:
    center_img, left_img,right_img,steering,throttle,brake, speed = l.split(", ")
    center_img = '/'.join(center_img.split('/')[-3:])
    left_img = '/'.join(left_img.split('/')[-3:])
    right_img  = '/'.join(right_img.split('/')[-3:])
    new_driving_log.append((center_img,steering, throttle, brake, speed))
    new_driving_log.append((left_img, steering, throttle, brake, speed))
    new_driving_log.append((right_img, steering, throttle, brake,speed))


with open("udacity_data/driving_log.pickle", "wb") as f:
    pickle.dump(new_driving_log,f)

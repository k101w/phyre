import math
import random
import pdb
#export MPLBACKEND=TKAgg
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook
#from celluloid import Camera
import phyre
import csv
import os

def data_pic(tasks,simulator):
      for task_index in range(len(tasks)):
            task_id = simulator.task_ids[task_index]
            imgpath=os.path.join('datasets_pic',task_id)
            os.mkdir(imgpath)
            #pdb.set_trace()
            #TODO: save the initial scene
            #initial_scene = simulator.initial_scenes[task_index]
            # plt.imshow(phyre.observations_to_float_rgb(initial_scene))
            # print(initial_scene.shape)
            # plt.title(f'Task {task_index}');
            # plt.savefig('pic/ini{}.png'.format(task_index))

            # initial_featurized_objects = simulator.initial_featurized_objects[task_index]
            # print('Initial featurized objects shape=%s dtype=%s' % (initial_featurized_objects.features.shape, initial_featurized_objects.features.dtype))
            # bar=np.array([x for x in initial_featurized_objects.features[0] if x[5]==1])
            # bar=bar.reshape(1,bar.shape[0],-1)
            # ball=np.array([x for x in initial_featurized_objects.features[0] if x[4]==1])
            # ball=ball.reshape(1,ball.shape[0],-1)

            # bar=[x for x in initial_featurized_objects.features[0] if x[5]==1]
            # bar=bar.reshape(1,bar.shape[0],-1)
            # plt.imshow(phyre.observations_to_float_rgb(bar))
            # plt.savefig('ini_bar.png')
            # bar=[x for x in initial_featurized_objects.features[0] if x[5]==1]
            # bar=bar.reshape(1,bar.shape[0],-1)
            # plt.imshow(phyre.observations_to_float_rgb(bar))
            # plt.savefig('ini_bar.png')

            np.set_printoptions(precision=3)
            #print(initial_featurized_objects.features)

            actions = simulator.build_discrete_action_space(max_actions=1000)#escape the invalid situation
            #print('A random action:', actions[0])
            #print(actions)
            #pdb.set_trace()
            # The simulator takes an index into simulator.task_ids.
            #action = random.choice(actions)
            # Set need_images=False and need_featurized_objects=False to speed up simulation, when only statuses are needed.
            num=0
            try:
                  for i in range(len(actions)):
                        action=actions[i]
                        simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True,stride=60)
                        if(simulation.status==0):continue
                        img_ac=os.imgpath.join(imgpath,'act{}'.format(num))
                        num+=1
                        os.mkdir(img_ac)   
                        for j,image in enumerate(simulation.images):
                              img = phyre.observations_to_float_rgb(image)
                              plt.imsave('{}/img{}.jpg'.format(img_ac,j),img)
                        #plt.savefig('{}/act{}_{}.jpg'.format(imgpath,action,j))
                  #if(simulation.status!=0): break
            # May call is_* methods on the status to check the status.
            except(TypeError):
                  pdb.set_trace()
def datasets(tasks,simulator):
      for task_index in range(len(tasks)):
            task_id = simulator.task_ids[task_index]
            task_id=list(task_id)
            for i in range(len(task_id)):
                  if task_id[i] == ':':
                        task_id[i] = '.'
            task_id=''.join(task_id)
                        

            imgpath=os.path.join('datasets_img',task_id)
            os.makedirs(imgpath)
            vecpath=os.path.join('datasets_vec',task_id)
            os.makedirs(vecpath)
            actions = simulator.build_discrete_action_space(max_actions=1000)
            num=0
            try:
                  for i in range(len(actions)):
                        action=actions[i]
                        simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True,stride=60)
                        if(simulation.status==0):continue
                        img_ac=os.path.join(imgpath,'act{}'.format(num))
                        vec_ac=os.path.join(vecpath,'act{}'.format(num))
                        num+=1
                        os.mkdir(img_ac)   
                        os.mkdir(vec_ac)
                        fo=simulation.featurized_objects   
                        for j,image in enumerate(simulation.images):
                              img = phyre.observations_to_float_rgb(image)
                              plt.imsave('{}/img{}.jpg'.format(img_ac,j),img)
                        
                              filename = '{}/vec{}.csv'.format(vec_ac,j)
                              with open (filename,'w') as file_object:
                                    writer=csv.DictWriter(file_object,['x','y','angle','diameter','shape','color'])
                                    writer.writeheader()
                                    for k in range(fo.num_objects):
                                          writer.writerow({'x':fo.states[j][k][0],'y':fo.states[j][k][1],'angle':fo.states[j][k][2],'diameter':fo.diameters[k],'shape':fo.shapes[k],'color':fo.colors[k]})


                        #plt.savefig('{}/act{}_{}.jpg'.format(imgpath,action,j))
                  #if(simulation.status!=0): break
            # May call is_* methods on the status to check the status.
            except(TypeError):
                  pdb.set_trace()


      # filename = 'featurized_objects.csv'
      # with open (filename,'w') as file_object:
      #       writer=csv.writer(file_object)
      #       writer.writerow([simulation.featurized_objects.num_objects])
      #       writer.writerow([simulation.featurized_objects.num_scene_objects])
      #       writer.writerow([simulation.featurized_objects.num_user_inputs])
      #       writer.writerow([simulation.featurized_objects.colors])
      #       writer.writerow([simulation.featurized_objects.diameters])
      #       writer.writerow([simulation.featurized_objects.states[0]])
      #       for i in range((simulation.featurized_objects.features).shape[0]):
      #             for j in range((simulation.featurized_objects.features).shape[1]):
      #                   writer.writerow(simulation.featurized_objects.features[i][j])
      # print('Number of observations returned by simulator:', len(simulation.images))
      # #print(len(simulation.featurized_objects))
      # print(simulation.featurized_objects.shapes)
      # print(simulation.featurized_objects.diameters)
      # print(simulation.featurized_objects.states.shape)
      # print(simulation.featurized_objects.states.shape[0])
      # pdb.set_trace()
      # num_across = 5
      # height = int(math.ceil(len(simulation.images) / num_across))
      # fig, axs = plt.subplots(height, num_across, figsize=(20, 15))
      # fig.tight_layout()
      # plt.subplots_adjust(hspace=0.2, wspace=0.2)
def mk_gif(simulation):
      fig=plt.figure()
      camera=Camera(fig)
      for i,image in enumerate(simulation.images):
            img = phyre.observations_to_float_rgb(image)
            plt.imshow(img)
            camera.snap()

      animation = camera.animate()
      animation.save('pic/try{}.gif'.format(task_index),writer='pillow')
      # We can visualize the simulation at each timestep.
      # for i, (ax, image) in enumerate(zip(axs.flatten(), simulation.images)):
      #     # Convert the simulation observation to images.
      #     img = phyre.observations_to_float_rgb(image)
      #     ax.imshow(img)
      #     ax.title.set_text(f'Timestep {i}')
      #     ax.get_xaxis().set_ticks([])
      #     ax.get_yaxis().set_ticks([])
      #     ax.figure.savefig('{}.png'.format(i))

random.seed(0)
eval_setup = 'ball_cross_template'
# fold_id = 0  # For simplicity, we will just use one fold for evaluation.
# train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
# print(*dev_tasks, sep=', ')
# pdb.set_trace()
action_tier = phyre.eval_setup_to_action_tier(eval_setup)
# print('Action tier for', eval_setup, 'is', action_tier)
# tasks = dev_tasks[:10]
#tasks=['00123:097','00013:020','00016:194','00021:024','00111:023','00112:007']
tasks=['00000:000']

# Create the simulator from the tasks and tier.
simulator = phyre.initialize_simulator(tasks, action_tier)
task_index = 0  # Note, this is a integer index of task within simulator.task_ids.
datasets(tasks,simulator)
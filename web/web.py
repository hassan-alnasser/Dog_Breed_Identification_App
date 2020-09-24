import os
import cv2
from keras.models import load_model
from keras.preprocessing import image 
import numpy as np
from glob import glob
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from cv2 import cvtColor,CascadeClassifier,COLOR_BGR2GRAY,imread
from keras.applications.xception import Xception, preprocess_input

class DogModel:

  model = None
  graph = None
 
  def __init__(self):         
    
    self.model = load_model('model/weights.best.model.hdf5')            
    self.model_include_top_false = Xception(weights='imagenet',include_top=False)      
    self.this_model = Xception(weights='imagenet')    
    self.graph = tf.get_default_graph()    
    
    self.dog_names=['Affenpinscher',
                    'Afghan_hound',
                    'Airedale_terrier',
                    'Akita',
                    'Alaskan_malamute',
                    'American_eskimo_dog',
                    'American_foxhound',
                    'American_staffordshire_terrier',
                    'American_water_spaniel',
                    'Anatolian_shepherd_dog',
                    'Australian_cattle_dog',
                    'Australian_shepherd',
                    'Australian_terrier',
                    'Basenji',
                    'Basset_hound',
                    'Beagle',
                    'Bearded_collie',
                    'Beauceron',
                    'Bedlington_terrier',
                    'Belgian_malinois',
                    'Belgian_sheepdog',
                    'Belgian_tervuren',
                    'Bernese_mountain_dog',
                    'Bichon_frise',
                    'Black_and_tan_coonhound',
                    'Black_russian_terrier',
                    'Bloodhound',
                    'Bluetick_coonhound',
                    'Border_collie',
                    'Border_terrier',
                    'Borzoi',
                    'Boston_terrier',
                    'Bouvier_des_flandres',
                    'Boxer',
                    'Boykin_spaniel',
                    'Briard',
                    'Brittany',
                    'Brussels_griffon',
                    'Bull_terrier',
                    'Bulldog',
                    'Bullmastiff',
                    'Cairn_terrier',
                    'Canaan_dog',
                    'Cane_corso',
                    'Cardigan_welsh_corgi',
                    'Cavalier_king_charles_spaniel',
                    'Chesapeake_bay_retriever',
                    'Chihuahua',
                    'Chinese_crested',
                    'Chinese_shar-pei',
                    'Chow_chow',
                    'Clumber_spaniel',
                    'Cocker_spaniel',
                    'Collie',
                    'Curly-coated_retriever',
                    'Dachshund',
                    'Dalmatian',
                    'Dandie_dinmont_terrier',
                    'Doberman_pinscher',
                    'Dogue_de_bordeaux',
                    'English_cocker_spaniel',
                    'English_setter',
                    'English_springer_spaniel',
                    'English_toy_spaniel',
                    'Entlebucher_mountain_dog',
                    'Field_spaniel',
                    'Finnish_spitz',
                    'Flat-coated_retriever',
                    'French_bulldog',
                    'German_pinscher',
                    'German_shepherd_dog',
                    'German_shorthaired_pointer',
                    'German_wirehaired_pointer',
                    'Giant_schnauzer',
                    'Glen_of_imaal_terrier',
                    'Golden_retriever',
                    'Gordon_setter',
                    'Great_dane',
                    'Great_pyrenees',
                    'Greater_swiss_mountain_dog',
                    'Greyhound',
                    'Havanese',
                    'Ibizan_hound',
                    'Icelandic_sheepdog',
                    'Irish_red_and_white_setter',
                    'Irish_setter',
                    'Irish_terrier',
                    'Irish_water_spaniel',
                    'Irish_wolfhound',
                    'Italian_greyhound',
                    'Japanese_chin',
                    'Keeshond',
                    'Kerry_blue_terrier',
                    'Komondor',
                    'Kuvasz',
                    'Labrador_retriever',
                    'Lakeland_terrier',
                    'Leonberger',
                    'Lhasa_apso',
                    'Lowchen',
                    'Maltese',
                    'Manchester_terrier',
                    'Mastiff',
                    'Miniature_schnauzer',
                    'Neapolitan_mastiff',
                    'Newfoundland',
                    'Norfolk_terrier',
                    'Norwegian_buhund',
                    'Norwegian_elkhound',
                    'Norwegian_lundehund',
                    'Norwich_terrier',
                    'Nova_scotia_duck_tolling_retriever',
                    'Old_english_sheepdog',
                    'Otterhound',
                    'Papillon',
                    'Parson_russell_terrier',
                    'Pekingese',
                    'Pembroke_welsh_corgi',
                    'Petit_basset_griffon_vendeen',
                    'Pharaoh_hound',
                    'Plott',
                    'Pointer',
                    'Pomeranian',
                    'Poodle',
                    'Portuguese_water_dog',
                    'Saint_bernard',
                    'Silky_terrier',
                    'Smooth_fox_terrier',
                    'Tibetan_mastiff',
                    'Welsh_springer_spaniel',
                    'Wirehaired_pointing_griffon',
                    'Xoloitzcuintli',
                    'Yorkshire_terrier']     
   
  def predict(self,img_path):
    # extract bottleneck features
    bottleneck_feature = self.extract_Xception(self.path_to_tensor(img_path))
    # obtain predicted vector
    with self.graph.as_default():
        predicted_vector =  self.model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return self.dog_names[np.argmax(predicted_vector)]

  def extract_Xception(self,tensor):
    with self.graph.as_default():
        return self.model_include_top_false.predict(preprocess_input(tensor))

  def path_to_tensor(self,img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0) 

  # returns "True" if face is detected in image stored at img_path
  def face_detector(self,img_path):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    img = imread(img_path)
    gray = cvtColor(img, COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
  
  def predict_labels(self,img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(self.path_to_tensor(img_path))
    with self.graph.as_default():
        return np.argmax(self.this_model.predict(img))
  
  def dog_detector(self,img_path):
    prediction = self.predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))
  
  def detect_dog_human(self,img_path):
    isDog = self.dog_detector(img_path)
    if isDog:
        retVal =  ['Detected as a dog', 'The predecited dog breed is: ',self.predict(img_path)] 
        return retVal
    else:
        isHuman = self.face_detector(img_path)
        if isHuman:
            #return the resembling dog breed
            retVal = ['Detected as a human', 'The resembling dog breed is: ',self.predict(img_path)] 
            return retVal
        else:
            return 0

  


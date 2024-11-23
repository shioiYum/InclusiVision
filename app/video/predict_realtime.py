import os
import cv2
import shutil
import numpy as np
import joblib
import functools
import operator
import time
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16
from app import app

# Define paths and constants
BASE_PATH = './app/video/'
TRAIN_PATH = BASE_PATH + "data/training_data"
TEST_PATH = "./app/Media/"
MODEL_SAVE_PATH = BASE_PATH + 'model_final/'
BATCH_SIZE = 320
LEARNING_RATE = 0.0007
EPOCHS = 150
LATENT_DIM = 512
ENCODER_TOKENS = 4096
DECODER_TOKENS = 1500
ENCODER_TIME_STEPS = 80
VALIDATION_SPLIT = 0.15
MAX_LENGTH = 10
SEARCH_METHOD = 'greedy'

def load_cnn_model():
    base_model = VGG16(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
    output_layer = base_model.layers[-2].output
    model = Model(inputs=base_model.input, outputs=output_layer)
    return model

def extract_frames(video_file):
    frames_dir = os.path.join(TEST_PATH, 'temporary_images')
    video_path = os.path.join(TEST_PATH, 'video', video_file)
    
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)
    
    frame_list = []
    capture = cv2.VideoCapture(video_path)
    count = 0
    
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
        frame_path = os.path.join(frames_dir, f'frame{count}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_list.append(frame_path)
        count += 1
    
    capture.release()
    cv2.destroyAllWindows()
    return frame_list

def extract_features_from_video(video_file, cnn_model):
    frame_list = extract_frames(video_file)
    sampled_frames = np.round(np.linspace(0, len(frame_list) - 1, ENCODER_TIME_STEPS))
    selected_frames = [frame_list[int(idx)] for idx in sampled_frames]
    
    image_data = np.zeros((len(selected_frames), 224, 224, 3))
    for idx, frame_path in enumerate(selected_frames):
        image_data[idx] = load_and_preprocess_image(frame_path)
    
    features = cnn_model.predict(image_data, batch_size=128)
    shutil.rmtree(os.path.join(TEST_PATH, 'temporary_images'))
    return features

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (224, 224))
    return resized_image

def load_inference_models():
    with open(os.path.join(MODEL_SAVE_PATH, f'tokenizer{DECODER_TOKENS}'), 'rb') as tokenizer_file:
        tokenizer = joblib.load(tokenizer_file)
    
    encoder_model = load_model(os.path.join(MODEL_SAVE_PATH, 'encoder_model.h5'))
    
    decoder_inputs = Input(shape=(None, DECODER_TOKENS))
    lstm_layer = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
    dense_layer = Dense(DECODER_TOKENS, activation='softmax')
    
    state_input_h = Input(shape=(LATENT_DIM,))
    state_input_c = Input(shape=(LATENT_DIM,))
    
    decoder_states_inputs = [state_input_h, state_input_c]
    lstm_outputs, state_h, state_c = lstm_layer(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_outputs = dense_layer(lstm_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs, state_h, state_c])
    
    decoder_model.load_weights(os.path.join(MODEL_SAVE_PATH, 'decoder_model_weights.h5'))
    return tokenizer, encoder_model, decoder_model

class RealTimeVideoCaptioning:
    def __init__(self):
        self.latent_dim = LATENT_DIM
        self.encoder_tokens = ENCODER_TOKENS
        self.decoder_tokens = DECODER_TOKENS
        self.encoder_time_steps = ENCODER_TIME_STEPS
        self.max_probability = -1
        self.search_method = SEARCH_METHOD
        self.current_video_index = 0
        self.tokenizer, self.encoder_model, self.decoder_model = load_inference_models()

    def greedy_search_caption(self, features):
        word_map = self.token_to_index_map()
        states = self.encoder_model.predict(features.reshape(-1, self.encoder_time_steps, self.encoder_tokens))
        
        target_seq = np.zeros((1, 1, self.decoder_tokens))
        target_seq[0, 0, self.tokenizer.word_index['bos']] = 1
        generated_caption = ''
        
        for _ in range(MAX_LENGTH):
            predictions, h, c = self.decoder_model.predict([target_seq] + states)
            states = [h, c]
            predicted_token = np.argmax(predictions.flatten())
            if predicted_token == 0 or word_map.get(predicted_token) in {'eos', None}:
                break
            generated_caption += word_map[predicted_token] + ' '
            target_seq = np.zeros((1, 1, self.decoder_tokens))
            target_seq[0, 0, predicted_token] = 1
        return generated_caption.strip()

    def token_to_index_map(self):
        return {index: token for token, index in self.tokenizer.word_index.items()}

    def get_test_video_features(self):
        video_list = os.listdir(os.path.join(TEST_PATH, 'video'))
        current_video = video_list[self.current_video_index]
        
        features_path = os.path.join(TEST_PATH, 'feat', f'{current_video}.npy')
        if os.path.exists(features_path):
            features = np.load(features_path)
        else:
            cnn_model = load_cnn_model()
            features = extract_features_from_video(current_video, cnn_model)
        
        self.current_video_index = (self.current_video_index + 1) % len(video_list)
        return features, current_video

    def generate_caption(self):
        features, video_name = self.get_test_video_features()
        if self.search_method == 'greedy':
            return self.greedy_search_caption(features.reshape(-1, self.encoder_time_steps, self.encoder_tokens)), video_name
        else:
            # Implement beam search logic here if needed
            pass

def caption_video():
    video_captioner = RealTimeVideoCaptioning()
    start_time = time.time()
    caption, video_file = video_captioner.generate_caption()
    elapsed_time = time.time() - start_time
    return caption, elapsed_time

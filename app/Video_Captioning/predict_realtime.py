import functools
import operator
import os
import cv2
import time
import shutil
import numpy as np
import joblib
from keras.layers import Input, LSTM, Dense
from tensorflow.keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from app import app

MAIN_PATH = './app/Video_Captioning/'

train_path = MAIN_PATH + "data/training_data"
test_path = "./app/Media/"
batch_size = 320
learning_rate = 0.0007
epochs = 150
latent_dim = 512
num_encoder_tokens = 4096
num_decoder_tokens = 1500
time_steps_encoder = 80
max_probability = -1
save_model_path = MAIN_PATH + 'model_final/'
validation_split = 0.15
max_length = 10
search_type = 'greedy'


def model_cnn_load():
    model = VGG16(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
    out = model.layers[-2].output
    model_final = Model(inputs=model.input, outputs=out)
    return model_final


def video_to_frames(video):
    path = os.path.join(test_path, 'temporary_images')
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    video_path = os.path.join(test_path, 'video', video)
    count = 0
    image_list = []
    # Path to video file
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        cv2.imwrite(os.path.join(test_path, 'temporary_images', 'frame%d.jpg' % count), frame)
        image_list.append(os.path.join(test_path, 'temporary_images', 'frame%d.jpg' % count))
        count += 1

    cap.release()
    cv2.destroyAllWindows()
    return image_list



def extract_features(video, model):
    """

    :param video: The video whose frames are to be extracted to convert into a numpy array
    :param model: the pretrained vgg16 model
    :return: numpy array of size 4096x80
    """
    video_id = video.split(".")[0]
    print(video_id)
    print(f'Processing video {video}')

    image_list = video_to_frames(video)
    samples = np.round(np.linspace(
        0, len(image_list) - 1, 80))
    image_list = [image_list[int(sample)] for sample in samples]
    images = np.zeros((len(image_list), 224, 224, 3))
    for i in range(len(image_list)):
        img = load_image(image_list[i])
        images[i] = img
    images = np.array(images)
    fc_feats = model.predict(images, batch_size=128)
    img_feats = np.array(fc_feats)
    # cleanup
    shutil.rmtree(os.path.join(test_path, 'temporary_images'))
    return img_feats


def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    return img


def inference_model():
    """Returns the model that will be used for inference"""
    print(os.getcwd())
    with open(os.path.join(save_model_path + 'tokenizer' + str(num_decoder_tokens)), 'rb') as file:
        tokenizer = joblib.load(file)
    # loading encoder model. This remains the same
    inf_encoder_model = load_model(os.path.join(save_model_path, 'encoder_model.h5'))

    # inference decoder model loading
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    inf_decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    inf_decoder_model.load_weights(os.path.join(save_model_path, 'decoder_model_weights.h5'))
    return tokenizer, inf_encoder_model, inf_decoder_model


class VideoDescriptionRealTime(object):
    """
        Initialize the parameters for the model
        """
    def __init__(self):
        self.latent_dim = latent_dim
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.time_steps_encoder = time_steps_encoder
        self.max_probability = max_probability

        # models
        self.tokenizer, self.inf_encoder_model, self.inf_decoder_model = inference_model()
        self.save_model_path = save_model_path
        self.test_path = test_path
        self.search_type = search_type
        self.num = 0

    def greedy_search(self, loaded_array):
        """

        :param f: the loaded numpy array after creating videos to frames and extracting features
        :return: the final sentence which has been predicted greedily
        """
        inv_map = self.index_to_word()
        states_value = self.inf_encoder_model.predict(loaded_array.reshape(-1, 80, 4096))
        target_seq = np.zeros((1, 1, 1500))
        final_sentence = ''
        target_seq[0, 0, self.tokenizer.word_index['bos']] = 1
        for i in range(15):
            output_tokens, h, c = self.inf_decoder_model.predict([target_seq] + states_value)
            states_value = [h, c]
            output_tokens = output_tokens.reshape(self.num_decoder_tokens)
            y_hat = np.argmax(output_tokens)
            if y_hat == 0:
                continue
            if inv_map[y_hat] is None:
                break
            if inv_map[y_hat] == 'eos':
                break
            else:
                final_sentence = final_sentence + inv_map[y_hat] + ' '
                target_seq = np.zeros((1, 1, 1500))
                target_seq[0, 0, y_hat] = 1
        return final_sentence

    def decode_sequence2bs(self, input_seq):
        states_value = self.inf_encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.tokenizer.word_index['bos']] = 1
        self.beam_search(target_seq, states_value, [], [], 0)
        return decode_seq

    def beam_search(self, target_seq, states_value, prob, path, lens):
        """

        :param target_seq: the array that is fed into the model to predict the next word
        :param states_value: previous state that is fed into the lstm cell
        :param prob: probability of predicting a word
        :param path: list of words from each sentence
        :param lens: number of words
        :return: final sentence
        """
        global decode_seq
        node = 2
        output_tokens, h, c = self.inf_decoder_model.predict(
            [target_seq] + states_value)
        output_tokens = output_tokens.reshape(self.num_decoder_tokens)
        sampled_token_index = output_tokens.argsort()[-node:][::-1]
        states_value = [h, c]
        for i in range(node):
            if sampled_token_index[i] == 0:
                sampled_char = ''
            else:
                sampled_char = list(self.tokenizer.word_index.keys())[
                    list(self.tokenizer.word_index.values()).index(sampled_token_index[i])]
            MAX_LEN = 12
            if sampled_char != 'eos' and lens <= MAX_LEN:
                p = output_tokens[sampled_token_index[i]]
                if sampled_char == '':
                    p = 1
                prob_new = list(prob)
                prob_new.append(p)
                path_new = list(path)
                path_new.append(sampled_char)
                target_seq = np.zeros((1, 1, self.num_decoder_tokens))
                target_seq[0, 0, sampled_token_index[i]] = 1.
                self.beam_search(target_seq, states_value, prob_new, path_new, lens + 1)
            else:
                p = output_tokens[sampled_token_index[i]]
                prob_new = list(prob)
                prob_new.append(p)
                p = functools.reduce(operator.mul, prob_new, 1)
                if p > self.max_probability:
                    decode_seq = path
                    self.max_probability = p

    def decoded_sentence_tuning(self, decoded_sentence):
        # tuning sentence
        decode_str = []
        filter_string = ['bos', 'eos']
        uni_gram = {}
        last_string = ""
        for idx2, c in enumerate(decoded_sentence):
            if c in uni_gram:
                uni_gram[c] += 1
            else:
                uni_gram[c] = 1
            if last_string == c and idx2 > 0:
                continue
            if c in filter_string:
                continue
            if len(c) > 0:
                decode_str.append(c)
            if idx2 > 0:
                last_string = c
        return decode_str

    def index_to_word(self):
        # inverts word tokenizer
        index_to_word = {value: key for key, value in self.tokenizer.word_index.items()}
        return index_to_word

    def get_test_data(self):
        # loads the features array
        file_list = os.listdir(os.path.join(self.test_path, 'video'))
        # with open(os.path.join(self.test_path, 'testing.txt')) as testing_file:
            # lines = testing_file.readlines()
        # file_name = lines[self.num].strip()
        file_name = file_list[self.num]
        path = os.path.join(self.test_path, 'feat', file_name + '.npy')
        if os.path.exists(path):
            f = np.load(path)
        else:
            model = model_cnn_load()
            f = extract_features(file_name, model)
        if self.num < len(file_list):
            self.num += 1
        else:
            self.num = 0
        return f, file_name

    def test(self):
        X_test, filename = self.get_test_data()
        # generate inference test outputs
        if self.search_type == 'greedy':
            sentence_predicted = self.greedy_search(X_test.reshape((-1, 80, 4096)))
        else:
            sentence_predicted = ''
            decoded_sentence = self.decode_sequence2bs(X_test.reshape((-1, 80, 4096)))
            decode_str = self.decoded_sentence_tuning(decoded_sentence)
            for d in decode_str:
                sentence_predicted = sentence_predicted + d + ' '
        # re-init max prob
        self.max_probability = -1
        return sentence_predicted, filename

    def main(self, filename, caption):
        """

        :param filename: the video to load
        :param caption: final caption
        :return:
        """
        # 1. Initialize reading video object
        cap1 = cv2.VideoCapture(os.path.join(self.test_path, 'video', filename))
        cap2 = cv2.VideoCapture(os.path.join(self.test_path, 'video', filename))
        caption = '[' + ' '.join(caption.split()[1:]) + ']'
        # 2. Cycle through pictures
        while cap1.isOpened():
            ret, frame = cap2.read()
            ret2, frame2 = cap1.read()
            if ret:
                imS = cv2.resize(frame, (480, 300))
                cv2.putText(imS, caption, (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                            2, cv2.LINE_4)
                cv2.imshow("VIDEO CAPTIONING", imS)
            if ret2:
                imS = cv2.resize(frame, (480, 300))
                cv2.imshow("ORIGINAL", imS)
            else:
                break

            # Quit playing
            key = cv2.waitKey(25)
            if key == 27:  # Button esc
                break

        # 3. Free resources
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()


def predict_video():
    video_to_text = VideoDescriptionRealTime()
    print('.........................\nGenerating Caption:\n')
    start = time.time()
    video_caption, file = video_to_text.test()
    end = time.time()
    return video_caption

import json
import logging
import os

import torch
from rnn import RNNModel

import data

# Make call like this:
# predictor.predict({"words": "My words"})

JSON_CONTENT_TYPE = 'application/json'

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    # def meth():
    #     return "Meth A"
    return "modelstuff"


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data
    raise Exception(
        'Requested unsupported ContentType in content_type: ' + content_type)


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


def predict_fn(input_data, model_fn_output):
    logger.info("Input data ", input_data)
    output = "HELLO!!! " + model_fn_output + input_data["words"]
    logger.info("Output data ", output)
    return output
    # logger.info('Generating text based on input parameters.')
    # corpus = model['corpus']
    # model = model['model']

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logger.info('Current device: {}'.format(device))
    # torch.manual_seed(input_data['seed'])
    # ntokens = len(corpus.dictionary)
    # input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
    # hidden = model.init_hidden(1)

    # logger.info('Generating {} words.'.format(input_data['words']))
    # result = []
    # with torch.no_grad():  # no tracking history
    #     for i in range(input_data['words']):
    #         output, hidden = model(input, hidden)
    #         word_weights = output.squeeze().div(
    #             input_data['temperature']).exp().cpu()
    #         word_idx = torch.multinomial(word_weights, 1)[0]
    #         input.fill_(word_idx)
    #         word = corpus.dictionary.idx2word[word_idx]
    #         word = word if type(word) == str else word.decode()
    #         if word == '<eos>':
    #             word = '\n'
    #         elif i % 12 == 11:
    #             word = word + '\n'
    #         else:
    #             word = word + ' '
    #         result.append(word)
    # return ''.join(result)

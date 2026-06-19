# Miscellaneous functions.


import os
import numpy as np

def export_input_arguments(param, save_path):

    number_file = 0
    file_parameters_fullpath = os.path.join(save_path, 'input_parameters.txt')

    while (os.path.exists(file_parameters_fullpath)):
        file_parameters_fullpath = os.path.join(save_path, 'input_parameters_new_' + str(number_file) + '.txt')
        number_file = number_file + 1

    file_text = open(file_parameters_fullpath, mode='a+')


    file_text.write('################### BEGINNING OF PARAMETERS ###################\n')

    for name_parameter, value in param.items():

        value_txt = ''

        if isinstance(value, str):
            if len(value) >=2:
                if (value[0] == '#' or value[-1] == '#'):
                    file_text.write(value + '\n')

                else:
                    value_txt = value
                    file_text.write(name_parameter + ' = ' + value_txt + '\n')

            else:
                value_txt = value
                file_text.write(name_parameter + ' = ' + value_txt + '\n')

        else:

            if isinstance(value, str):

                value_txt = value

            elif isinstance(value, int) or isinstance(value, float) or isinstance(value, bool) or isinstance(value, type(None)):


                value_txt = str(value)

            # Assuming a list of numbers
            elif isinstance(value, list):

                value_txt = '[ '

                for value_index, value_element in enumerate(value):
                    value_txt = value_txt + str(value_element) + ' '

                value_txt = value_txt + ']'

            # Assuming a tuple of numbers
            elif isinstance(value, tuple):

                value_txt = '( '

                for value_index, value_element in enumerate(value):
                    value_txt = value_txt + str(value_element) + ' '

                value_txt = value_txt + ')'

            # Assuming a range of numbers
            elif isinstance(value, range):


                file_text.write('VERY IMPORTANT NOTE: THE LAST VALUE HERE IS INCLUDED, CONTRARILY TO PYTHON RANGES WRITTEN INSIDE PROGRAM\n')
                value_txt = '(' + str(value[0]) + ', ' + str(value[-1]) + ', ' + str(value.step) + ')'

            elif isinstance(value, dict):
                value_txt = '{ '

                for dict_key, dict_value in value.items():
                    value_txt = value_txt + str(dict_key) + ': ' + str(dict_value) + ' '

                value_txt = value_txt + '}'

            elif isinstance(value, np.ndarray):

                file_text.write('NOTATION: (initial_value (included), last_value (included), number_of_elements)\n')
                value_txt = '(' + str(np.min(value)) + ', ' + str(np.max(value)) + ', ' + str(len(value)) + ')'

            file_text.write(name_parameter + ' = ' + value_txt + '\n')




    file_text.write('################### END OF PARAMETERS ###################\n')

    file_text.close()

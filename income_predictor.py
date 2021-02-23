"""
Using a dataset (the "Adult Data Set") from the UCI Machine-Learning Repository we can predict based on a number
of factors whether or not someone's income will be greater than $50,000.

The technique:

    The approach is to create a 'classifier' - a program that takes a new example record and, based on previous
    examples, determines which 'class' it belongs to. In this problem we consider attributes of records and separate
    these into two broad classes, <=50K and >50K.
    We begin with a training data set - examples with known solutions. The classifier looks for patterns that indicate
    classification. These patterns can be applied against new data to predict outcomes. If we already know the outcomes
    of the test data, we can test the reliability of our model. if it proves reliable we could then use it to classify
    data with unknown outcomes.
    We must train the classifier to establish an internal model of the patterns that distinguish our two classes. Once
    trained we can apply this against the test data - which has known outcomes.
    We take our data and split it into two groups - training and test - with most of the data in the training set.
    We need to write a program to find the patterns in the training set.

Process overview:

    1) Create training set from data.
    2) Create classifier using training dataset to determine separator values for each attribute.
    3) Create test dataset.
    4) Use classifier to classify data in test set while maintaining accuracy score.

LAST MODIFIED - 12/12/2020
"""

import requests

ALLOWED_CONTENT_TYPES = ("application/x-httpd-php", "text/plain", "text/html")
DATA_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
PERCENT = 75


def print_error_and_exit(error, return_code):
    print(f"{'=' * 50}\nSomething bad happened.\n{error}\n{'=' * 50}")
    quit(return_code)


def get_file_from_net(url):
    try:
        response = requests.get(url)
        if 200 <= response.status_code <= 299:
            if response.headers["Content-Type"] and response.headers["Content-Type"] in ALLOWED_CONTENT_TYPES:
                return response.text
            else:
                raise ValueError(f"Doesnt look like printable content\nContent-Type is "
                                 f"{response.headers['Content-Type']}'")
        else:
            raise ValueError(f"Bad status code: {response.status_code}")
    except Exception as e:
        print_error_and_exit(e, 1)


def get_data(data):
    """
    Here we read our data file from the web and split it out into a list of tuples, one tuple per record.
        -   If record is incomplete (one or more attributes = '?') - we disregard the record and move to the next one.
        -   If record is empty - we output ValueError.
        -   Finally, we are deleting elements/attributes that are not required for our study.
    """
    bad_records = 0
    cleaned_dataset = []

    # using only .strip() doesnt remove all spaces before/after comma so additionally decided to use replace().
    data = data.replace(', ', ',').replace(' ,', ',').strip().split('\n')

    for record in data:
        try:
            if '?' not in record:
                record = record.split(",")
                if not record:
                    raise ValueError("Empty Record")
                del record[2:4], record[11]
                cleaned_dataset.append(record)
            else:
                continue

        except ValueError as val_err:
            bad_records += 1
            print(f"Record {record[0]} rejected: {val_err}")
            continue

    return tuple(cleaned_dataset)


def create_classifier(data):
    """
    Here we are creating classifier using 75% of records from cleaned dataset. Returning list of midpoint values for
    every numeric attribute + dictionary with every non-numeric attribute(key) and its outcome(value: pos/neg/neutral).
        -   Checking every record in the list if it is positive (>50K) or negative (<=50K).
        -   Then checking every element in the record if it is numeric or not, and separating them on the lists of
            Positive/Negative Numeric Attributes and Positive/Negative Not Numeric Attributes.
        -   Creating the unique list of all possible non-numeric attributes (not_num_sub_attr).
        -   Finally, finding midpoint for numeric attributes and outcomes for non-numeric attributes.

    IMPORTANT:  We know that in total we have 5 numeric attributes and 6 non-numeric attributes
                (excluding outcome attribute [-1]) and all of them repeated in the same sequence in each record.
                Therefore, after separating attributes on different lists we know when they will repeat in those lists.
    """
    pos_num_attr = []
    neg_num_attr = []
    midpoint_num = []

    pos_not_num_attr = []
    neg_not_num_attr = []
    not_num_sub_attr = []
    not_num_sub_attr_outcome = []

    for record in data:
        if record[-1] == '>50K':
            for i in range(0, len(record) - 1):
                if record[i].isnumeric():
                    pos_num_attr.append((int(record[i])))
                else:
                    pos_not_num_attr.append(record[i])

        else:
            for i in range(0, len(record) - 1):
                if record[i].isnumeric():
                    neg_num_attr.append((int(record[i])))
                else:
                    neg_not_num_attr.append(record[i])

        # Below loop checks every element in the record and if it is non-numeric and it is not already in the list
        # (not_num_sub_attr) - it will add it to that list.
        [not_num_sub_attr.append(record[attr]) for attr in range(0, len(record) - 1) if record[attr] not in
         not_num_sub_attr and record[attr].isnumeric() is False]

    # Finding the midpoints for numeric attributes (READ => IMPORTANT).
    for i in range(5):
        midpoint_num.append((sum(pos_num_attr[i::5]) / len(pos_num_attr[i::5]) +
                             sum(neg_num_attr[i::5]) / len(neg_num_attr[i::5])) / 2)

    # Finding the outcomes for non-numeric attributes (READ => IMPORTANT).
    for i in range(len(not_num_sub_attr)):
        pos_ave = pos_not_num_attr.count(not_num_sub_attr[i]) / len(pos_not_num_attr[::6])
        neg_ave = neg_not_num_attr.count(not_num_sub_attr[i]) / len(neg_not_num_attr[::6])
        if pos_ave > neg_ave:
            not_num_sub_attr_outcome.append('pos')
        elif pos_ave < neg_ave:
            not_num_sub_attr_outcome.append('neg')
        else:
            not_num_sub_attr_outcome.append('neutral')

    # Creating dict from the list of unique non-numeric attributes(keys) and the list of outcomes(values).
    not_num_attr_dict = dict(zip(not_num_sub_attr, not_num_sub_attr_outcome))

    return tuple(midpoint_num), not_num_attr_dict


def test_classifier(test_dataset, classifier):
    """
    Here we are testing our classifier by applying the classifier list and dict against each record in the test dataset
    ana returning accuracy score.
        -   We compare every numeric attribute against its equivalent value in the classifier list.
        -   We give every non-numeric attribute the value based on its equivalent key in the classifier dict.
        -   Based on this, all attributes are updating the positive and negative counter, which are then used to
            determine the record outcome (>50K or <=50K).
        -   Finally, the outcome for every record that was predicted using classifier is compared against the actual
            outcome of the record from test dataset and accuracy score is determined.
    """
    incorrect_classification = 0

    for record in test_dataset:
        pos_counter = 0
        neg_counter = 0
        next_index = 0  # Using this counter to step forward one position whenever next numeric attribute is compared

        for i in range(0, len(record) - 1):
            if record[i].isnumeric():
                if int(record[i]) > classifier[0][next_index]:
                    pos_counter += 1
                else:
                    neg_counter += 1
                next_index += 1
            else:
                if classifier[1].get(record[i]) == 'pos':
                    pos_counter += 1
                elif classifier[1].get(record[i]) == 'neg':
                    neg_counter += 1
                else:
                    continue

        if pos_counter > neg_counter:
            if record[-1] != '>50K':
                incorrect_classification += 1
            else:
                continue
        else:
            if record[-1] != '<=50K':
                incorrect_classification += 1
            else:
                continue

    accuracy_score = round(abs(incorrect_classification / len(test_dataset) * 100 - 100), 2)

    return accuracy_score


def main():
    net_content = get_file_from_net(DATA_URL)
    cleaned_dataset = get_data(net_content)
    training_dataset = cleaned_dataset[:int(len(cleaned_dataset) * PERCENT / 100)]
    test_dataset = cleaned_dataset[int(len(cleaned_dataset) * PERCENT / 100):]
    accuracy_score = test_classifier(test_dataset, create_classifier(training_dataset))

    print(f'TOTAL RECORDS (TEST DATASET): {len(test_dataset)} | CORRECT CLASSIFICATIONS: '
          f'{round(len(test_dataset) / 100 * accuracy_score)} | INCORRECT CLASSIFICATIONS: '
          f'{round(len(test_dataset) - len(test_dataset) / 100 * accuracy_score)} | '
          f'ACCURACY SCORE: {accuracy_score} %')


if __name__ == "__main__":
    main()

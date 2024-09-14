import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import nltk
from collections import defaultdict, Counter
from nltk.corpus import brown
from nltk.tokenize import word_tokenize


# Download the Brown corpus
nltk.download('brown')

nltk.download('universal_tagset')

nltk.download('punkt')

# Load the Brown corpus with universal tagset
tagged_sentences = brown.tagged_sents(tagset='universal')

# Extract words and tags
tagged_words = [(word.lower(), tag) for sentence in tagged_sentences for (word, tag) in sentence]
words, tags = zip(*tagged_words)

unique_tags = set(tags) # Get all the distinct tags present in the dataset
print("No. of unique tags in the Brown corpus: ", len(unique_tags))


#############################################
####### Probablity of P(tag_i | tag_i-1) ####
#############################################

tag_transition_prob = defaultdict(lambda: defaultdict(float))

# Calculate no. of possibilities for each sequence of 2 tags
for sentence in tagged_sentences:
    prev_tag = "START"
    for word, tag in sentence:
        tag_transition_prob[prev_tag][tag] += 1
        prev_tag = tag
    tag_transition_prob[prev_tag]["END"] += 1

# Calculate probabilities
for prev_tag in tag_transition_prob: # Loop over all tag i
    total_count = sum(tag_transition_prob[prev_tag].values())
    for tag in tag_transition_prob[prev_tag]:
        tag_transition_prob[prev_tag][tag] /= total_count

#############################################################
############## Calculate emission probabilities #############
#############################################################

# Calculate no. of words for every given tag
word_given_tag_prob = defaultdict(lambda: defaultdict(float))
for word, tag in tagged_words:
    word_given_tag_prob[tag][word] += 1

# Calculate probability from word counts for each tag
for tag in word_given_tag_prob:
    total_count = sum(word_given_tag_prob[tag].values())
    for word in word_given_tag_prob[tag]:
        word_given_tag_prob[tag][word] /= total_count


def viterbi(sentence, transition_probs, emission_probs, unique_tags):

    viterbi_matrix = [{}]
    last_best_tag = [{}]

    # Initialize with the start state using log probabilities
    for tag in unique_tags:
        transition_prob = np.log(transition_probs["START"].get(tag, 1e-6))
        emission_prob = np.log(emission_probs[tag].get(sentence[0].lower(), 1e-6))
        viterbi_matrix[0][tag] = transition_prob + emission_prob
        last_best_tag[0][tag] = "START"

    # Update the viterbi matrix elements for every word in the sentence using log probabilities
    for t in range(1, len(sentence)):
        viterbi_matrix.append({})
        last_best_tag.append({})

        for tag in unique_tags:
            max_prob, best_prev_tag = max(
                (viterbi_matrix[t-1][prev_tag] +
                 np.log(transition_probs[prev_tag].get(tag, 1e-6)) +
                 np.log(emission_probs[tag].get(sentence[t].lower(), 1e-6)), prev_tag)
                for prev_tag in unique_tags
            )
            viterbi_matrix[t][tag] = max_prob
            last_best_tag[t][tag] = best_prev_tag

    # Compute final state probabilities for different paths using log probabilities
    max_prob, best_prev_tag = max(
        (viterbi_matrix[-1][tag] + np.log(transition_probs[tag].get("END", 1e-6)), tag)
        for tag in unique_tags
    )

    # Get the final sequence of tags
    best_path = [best_prev_tag]
    for t in range(len(sentence)-1, 0, -1):
        best_prev_tag = last_best_tag[t][best_prev_tag]
        best_path.insert(0, best_prev_tag)

    return best_path



def demo_hmm_pos_tagger(sentence, transition_prob, emission_prob, unique_tags):
    tokens = word_tokenize(sentence)
    tokens_lower_case = [token.lower() for token in tokens]
    predicted_tags = viterbi(tokens_lower_case, transition_prob, emission_prob, unique_tags)
    output = []
    for token, tag in zip(tokens, predicted_tags):
        output.append((token, tag))
    return output

# Streamlit app UI
def main():
    st.title("Viterbi Algorithm for POS Tagging")

    # Input text from the user
    user_input = st.text_input("Enter a sentence:")

    if user_input.strip():
        output = demo_hmm_pos_tagger(user_input, tag_transition_prob, word_given_tag_prob, unique_tags)

        st.write('Prediction')
        st.write(output)

if __name__ == "__main__":
    main()

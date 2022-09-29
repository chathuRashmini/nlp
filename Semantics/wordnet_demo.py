# -*- coding: utf-8 -*-
"""
Created July 2017

@author: arw
"""

from nltk.corpus import wordnet as wn
import pandas as pd

# First let's see how to access the meanings (senses) of a word - 'fruit' in this case
# Try other words that you maybe interested in
term = 'fruit'
synsets = wn.synsets(term)

# How many different meanings (senses) does 'fruit' have? Surprised?
# English is 'notorious' for lexical ambiguity - what about Sinhala and Tamil?
print('Total Synsets:', len(synsets))

# Synsets for 'fruit'
for synset in synsets:
    print('Synset:', synset)
    print('Part of speech:', synset.lexname())
    print('Definition:', synset.definition())
    print('Lemmas:', synset.lemma_names())
    print('Examples:', synset.examples())
    print()




# Exploring lexical semantic relationships: 
# entailments, homonyms/homographs, synonyms/antonyms, hyponyms/hyponyms, holonyms/meronyms


# Entailments

for action in ['walk', 'eat', 'digest']:
    action_syn = wn.synsets(action, pos='v')[0]
    print(action_syn, '-- entails -->', action_syn.entailments())
    
    
# Homonyms\homographs  

for synset in wn.synsets('bank'):
    print(synset.name(),'-',synset.definition())
# One of the meanings of 'bank' is to 'tip laterally' - see what the example (gloss) for this is



# Synonyms and antonyms of 'large'

term = 'large'
synsets = wn.synsets(term)
# We just take the second meaning - 'large' used as an adjective
adj_large = synsets[1]
# And only the first lemma for that particular sense of 'large'
adj_large = adj_large.lemmas()[0]
adj_large_synonym = adj_large.synset()
# We also inspect the antonym (opposite sense) of this particular sense of 'large'
adj_large_antonym = adj_large.antonyms()[0].synset()
# Try other senses of 'large' and check the results

# We can print the definition and gloss (examples) to get an idea of the senses concerned
print('Synonym:', adj_large_synonym.name())
print('Definition:', adj_large_synonym.definition())
print('Definition:', adj_large_synonym.examples())
print('Antonym:', adj_large_antonym.name())
print('Definition:', adj_large_antonym.definition())
print('Definition:', adj_large_antonym.examples())
print()


# Let's inspect the first three senses of the word 'rich' - how many are there in WN?
term = 'rich'
synsets = wn.synsets(term)[:3]

for synset in synsets:
    rich = synset.lemmas()[0]
    rich_synonym = rich.synset()
    rich_antonym = rich.antonyms()[0].synset()
    print('Synonym:', rich_synonym.name())
    print('Definition:', rich_synonym.definition())
    print('Antonym:', rich_antonym.name())
    print('Definition:', rich_antonym.definition())
    print()



# Hyponyms and hypernyms (i.e. the 'isa' relationship)

term = 'tree'
synsets = wn.synsets(term)
tree = synsets[0]
# Print the entity and its meaning
print('Name:', tree.name())
print('Definition:', tree.definition())

# Hyponyms - more specialized forms of 'tree'
hyponyms = tree.hyponyms()
# Print all hyponyms and some sample hyponyms for 'tree' (how many are there?)
print('Total Hyponyms:', len(hyponyms))
print('Sample Hyponyms')
for hyponym in hyponyms[:10]:
    print(hyponym.name(), '-', hyponym.definition())
    print()

# Hypernyms - more general forms of 'tree'  
hypernyms = tree.hypernyms()
print(hypernyms)

# Let's generate the full hierarchy from 'entity' to 'tree'
hypernym_paths = tree.hypernym_paths()
print('Total Hypernym paths:', len(hypernym_paths))
# Print the entire hypernym hierarchy
print('Hypernym Hierarchy')
print(' -> '.join(synset.name() for synset in hypernym_paths[0]))




# Holonyms and meronyms

# Holonyms - what is the 'tree' part-of?
member_holonyms = tree.member_holonyms()    
print('Total Member Holonyms:', len(member_holonyms))
print('Member Holonyms for [tree]:-')
for holonym in member_holonyms:
    print(holonym.name(), '-', holonym.definition())
    print()

# Meronyms - what are the parts of 'tree' (sub-parts)
part_meronyms = tree.part_meronyms()
print('Total Part Meronyms:', len(part_meronyms))
print('Part Meronyms for [tree]:-')
for meronym in part_meronyms:
    print(meronym.name(), '-', meronym.definition())
    print()

# Substance based meronyms of 'tree'
substance_meronyms = tree.substance_meronyms()    
print('Total Substance Meronyms:', len(substance_meronyms))
print('Substance Meronyms for [tree]:-')
for meronym in substance_meronyms:
    print(meronym.name(), '-', meronym.definition())
    print()





# Semantic relationships and similarities

# First lets think of some common words
tree = wn.synset('tree.n.01')
lion = wn.synset('lion.n.01')
tiger = wn.synset('tiger.n.02')
cat = wn.synset('cat.n.01')
dog = wn.synset('dog.n.01')

# And extract their names and definitions from WN
entities = [tree, lion, tiger, cat, dog]
entity_names = [entity.name().split('.')[0] for entity in entities]
entity_definitions = [entity.definition() for entity in entities]

for entity, definition in zip(entity_names, entity_definitions):
    print(entity, '-', definition)
    print()

# We want to find out how closely they are related in the WN hierarchy
common_hypernyms = []
for entity in entities:
    # Get pairwise lowest common hypernyms
    common_hypernyms.append([entity.lowest_common_hypernyms(compared_entity)[0]
                                            .name().split('.')[0]
                             for compared_entity in entities])
# Build pairwise lower common hypernym matrix
common_hypernym_frame = pd.DataFrame(common_hypernyms,
                                     index=entity_names, 
                                     columns=entity_names)
# Print the matrix
print(common_hypernym_frame)

# SAQ: Check what happens if you add something like 'table' (furniture sense and data-table sense)


# We can quantify this as a similarity measure using 'path similarity' in WN
similarities = []
for entity in entities:
    # Get pairwise similarities
    similarities.append([round(entity.path_similarity(compared_entity), 2)
                         for compared_entity in entities])        
# Build pairwise similarity matrix                             
similarity_frame = pd.DataFrame(similarities,
                                index=entity_names, 
                                columns=entity_names)
# Print the matrix of scores
print(similarity_frame)
# Check if these (and others you try) are reasonable





# Word sense disambiguation (WSD)
from nltk.wsd import lesk
from nltk import word_tokenize

# Sample text and word to disambiguate
samples = [('The fruits on that plant have ripened', 'n'),
           ('He finally reaped the fruit of his hard work as he won the race', 'n')]
word = 'fruit'

# We use the well-known Lesk algorithm to help us disambiguate the two senses of 'fruit' above
for sentence, pos_tag in samples:
    word_syn = lesk(word_tokenize(sentence.lower()), word, pos_tag)
    print('Sentence:', sentence)
    print('Word synset:', word_syn)
    print('Corresponding defition:', word_syn.definition())
    print()

# We use another example for the word 'lead' which is ambiguous in more ways than one!
samples = [('Lead is a very soft, malleable metal', 'n'),
           ('John is the actor who plays the lead in that movie', 'n'),
           ('This road leads to nowhere', 'v')]
word = 'lead'

# We use the Lesk algorithm again to help us disambiguate the two senses of 'lead'
for sentence, pos_tag in samples:
    word_syn = lesk(word_tokenize(sentence.lower()), word, pos_tag)
    print('Sentence:', sentence)
    print('Word synset:', word_syn)
    print('Corresponding defition:', word_syn.definition())
    print()


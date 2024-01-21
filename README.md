# MemNN

This project is a first implementation of Memory Networks in Python as the final project of a Deep Learning class. This project is mainly inspired by aykutaaykut's git: https://github.com/aykutaaykut/Memory-Networks


MemNN have 4 components: 
- I component : converts input into inner feature representation.
- G component : saves the inner feature representation in the memory slot.
- O component : uses a scoring function to match between the question and sentences in the memory.
- R component : produces the final answer to the question.

The dataset is composed of tasks in English and questions with their answer and the index of the task giving the answer. Here's an example :

1 Mary moved to the bathroom.
2 John went to the hallway.
3 Where is Mary?        bathroom 1
4 Daniel went back to the hallway.
5 Sandra moved to the garden.
6 Where is Daniel?      hallway 4

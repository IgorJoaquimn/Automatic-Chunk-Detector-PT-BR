# Automatic Chunk Detector PT-BR
The objective of this project is to develop a chunk detector for use in a RAG app. The challenge arises when developing a RAG system that receives PT-BR YouTube subtitles as input, as there is no direct way to determine the boundaries of the chunks. The proposed solution is to implement a chunk detector using deep learning. The proposed approach involves the development of a system that is capable of identifying whether two sentences originate from the same paragraph. This system will subsequently be employed to segment the sentences into meaningful units, or "chunks".

## The Model

The ChunkDetector class is a neural network module designed to address the challenges of chunk detection in PT-BR. The model first compress the Bert Embeddings to a smaller vector space to process and compare sentences in context.

### Potency, Sign, and Value
The ChunkDetector uses three separate sequential neural network layers to process input text:

  * Potency: This layer consists of a linear transformation followed by a Sigmoid activation function. The Potency layer is used to determine the potential importance or weight of each part of the input text.
  * Sign: This layer includes a linear transformation followed by a Tanh activation function. The Sign layer evaluates the sentiment or polarity of the input text, providing information on whether the context and sentence exhibit a similar tone or attitude.
  * Value: This layer consists of a linear transformation followed by a ReLU activation function. The Value layer helps quantify the relative significance or magnitude of the input text.

By combining all three layers in different ways, it is possible to explore a wide range of patterns in the vector space. As an illustration, consider the following example:
For example:
  *     V = Potency(context) * Sign(next sentence) ; 
This equation takes the sentiment of the "next sentence" and scales it based on the significance present in the context.
  *     V = Value(context) * Sign(next sentence) + Potency(context) ;
This other equation takes the positive part of the context and scale it based on the significance of the "next sentence" while maintaining the original sentiment of the context.

Given the choice of equations, the classification is done using the Cosine Similarity of the vectors.

## The Data 

The Machado de Assis Corpus, a collection of Portuguese texts written in diverse formats, is currently being used to train the model. Each two sequential sentences in the same paragraph of each text is taken as one example to create the positive labels. In a similar manner, the negative samples are generated from random sentences from two distinct paragraphs.

## Contributions and Support

If you come across any issues, bugs, or have any suggestions for improvement, please feel free to report them in the project repository. You can also contribute through pull requests.


**Author:** Igor Joaquim da Silva Costa

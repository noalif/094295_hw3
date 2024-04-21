introduction:

In this exercise, you will experience using machine learning algorithms and statistical tools for network analysis.
Your main task will consist of the analysis, investigation and classification of a data set describing a network of academic citations.

the data:

The data set is a directed graph, representing a network of academic citations.
The graph contains about 100,000 vertices where each vertex in the graph represents an article and each directed arc between vertex A and vertex B represents that vertex
A quoted the top of B.
Each article is represented by a feature vector created by averaging all the word representations (created by the gram-skip model) in the abstract and the title
His.
Also, the data set contains for each article the year of its publication and a number representing the category to which it belongs.

task:

In recent years, the pace of article publications has increased and there is a need for automatic tools that sort and make the articles accessible to researchers.
As part of this goal, your task in this exercise is to predict the article categories.

Restrictions:

• Do not install additional libraries other than those contained in the environment (yml.environment) you received.
• Do not change the Dataset3HW class.
• The training must be in the Learning Transductive paradigm - information from all vertices is allowed to flicker, including from validation,
But it is forbidden to use validation labels during the learning process (of course, in the inference it is forbidden to use labels).
in general (.
• Do not train models in the py.predict script. You can save a pre-trained model in your repository and read it from
the script.
• You must not enrich the data set with external sources of information. You must provide all the training code for your model in
repository.

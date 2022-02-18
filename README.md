# MASKS's Tool
MASKS is a tool in order to verify multi-classifier with a predefined property. Using MASKS, satisfaction of the property for all classifiers would be checked. In other words, if the property $\varphi$ satisfied using knowledge of all agents, it would be the verified formula (property). Here, we use the operator $D_A \varphi$ from Public Announcement Logic to collect distributed knowledge of classifiers. 

Here, for more convenient, we developed python codes in three steps to run the tool. Each first two step could be replaced by self created ones. These two steps are developed to provide inputs for the MASKS tool. 

The first one is ``0-create\_model.py'' in which, classifiers would be created. After defining the target dataset, the input data would be collected using ``load\_input\_data'' function. The output of this function should be contains a set of train images; and it should be stored in ``train\_data'' variable. In this python file, the train data would be applied to train classifiers (i.e.,  ANNs). The architecture of the classifiers could be defined in ``define\_model'' function in ``build\_model.py''. The number of output classes (``no\_classes'') and the input shape (``input\_shape'') should be determined. The output would be multiple classifiers. These classifiers would be stored into the ``Models'' folder. The number of classifiers could be determined by ``model\_no'' variable.


In order to run ``0-create\_model.py'', following inputs are required:
\begin{itemize}
    \item The number of output classes:
    would be stored in ``no\_classes'',
    \item The dimension of input images:
    would be stored in ``input\_shape'',
    \item The number of agents to be created:
    would be stored in ``model\_no'',
    \item train and validation dataset: would be stored in ``train\_data'', ``validation\_data'' (validation could be ignored),
    \item The classifier architecture:
    could be defined in ``define\_model'' in ``build\_model.py''.
    \item The ``train\_df\_path'' should be set to be the train file path, if you using ``load\_input\_data.py'' for loading image files.
\end{itemize}


Next, using the stored model in the ``Models'' folder, ``1-Eval\_model.py'' could be executed to evaluate test inputs and their neighborhoods and manipulations. The set of neighborhoods and manipulations would be defined in \textit{python class} ``NeighMan'' (here is a set of noise on input image). The output of ``1-Eval\_model.py'' would be results of inputs, neighborhoods and manipulations  for each classifier in a \textit{numpy} file in ``Results'' folder.


In order to run ``1-Eval\_model.py'', following inputs are required:
\begin{itemize}
    \item Image dimension:  would be stored in ``img\_width'', ``img\_height'', ``img\_num\_channels'',
    \item The input images: would be stored in ``input\_test'',
    \item Outputs of test inputs (and its neighborhoods and manipulations) for each classifier: would be stored \textit{numpy} array in files ``Results'' folder,
    \item The number of neighborhoods and manipulations:
    would be stored in ``no\_classes'',
    \item Ordered correct labels of inputs: would be stored in ``target\_test''.
    \item The ``project\_path'', ``data\_dir'',  should be set to be the project path and test file path.
\end{itemize}


Then, using the output results in the ``Results'' folder, ``2-MASKS.py'' could be executed. Here, the correct output labels would be provided with the function ``load\_labels''. After execution of this python code, following results for one to the number of classifiers would be provided:
\begin{enumerate}
    \item ``agent\_counter'': number of agents,
    \item ``correct\_answer'': correct verified answers,
    \item ``wrong\_answer'': wrong verified answer,
    \item ``conflict\_answer'': unverified answers because of conflicting,
    \item ``correct\_assist'': unverified answers because more than one output classes are provided, but the correct answer is in the provided output classes,
    \item ``wrong\_assist'':  unverified answers because more than one output classes are provided, but the correct answer is not in the provided output classes.
\end{enumerate}
For further investigation, result of these multi-agent systems, for every input would be stored in ``Agents'' folder.


In order to run ``2-MASKS.py'', following inputs are required:
\begin{itemize}
    \item Outputs of test inputs (and its neighborhoods and manipulations) for each classifier: would be stored \textit{numpy} array in files ``Results'' folder.
    \item The number of output classes:
    would be stored in ``no\_classes''
    \item The number of neighborhoods and manipulations:
    would be stored in ``nei\_man\_no''
    \item Ordered correct labels of inputs: would be stored in ``target\_test''.
\end{itemize}

The tool and Modified Fashion MNIST, MNIST, and Fruit-360 could be found in https://github.com/iuwa/MASKS (FashionMNIST-MASKS.zip, MNIST-MASKS.zip, and Fruit-360-MASKS.zip).

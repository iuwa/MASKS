# MASKS's Tool
MASKS is a tool in order to verify multi-classifier with a predefined property. Using MASKS, satisfaction of the property for all classifiers would be checked. In other words, if the property $\varphi$ satisfied using knowledge of all agents, it would be the verified formula (property). Here, we use the operator $D_A \varphi$ from Public Announcement Logic to collect distributed knowledge of classifiers. 

Here, for more convenient, we developed python codes in three steps to run the tool. Each first two step could be replaced by self created ones. These two steps are developed to provide inputs for the MASKS tool. 

The first one is "0-create_model.py" in which, classifiers would be created. After defining the target dataset, the input data would be collected using ``load_input_data'' function. The output of this function should be contains a set of train images; and it should be stored in ``train_data'' variable. In this python file, the train data would be applied to train classifiers (i.e.,  ANNs). The architecture of the classifiers could be defined in ``define_model'' function in ``build_model.py''. The number of output classes (``no_classes'') and the input shape (``input_shape'') should be determined. The output would be multiple classifiers. These classifiers would be stored into the ``Models'' folder. The number of classifiers could be determined by ``model_no'' variable.


In order to run ``0-create_model.py'', following inputs are required:
* The number of output classes: would be stored in ``no_classes'',
* The dimension of input images: would be stored in ``input_shape'',
* The number of agents to be created: would be stored in ``model_no'',
* train and validation dataset: would be stored in ``train_data'', ``validation_data'' (validation could be ignored),
* The classifier architecture: could be defined in ``define_model'' in ``build_model.py''.
* The ``train_df_path'' should be set to be the train file path, if you using ``load_input_data.py'' for loading image files.



Next, using the stored model in the ``Models'' folder, ``1-Eval_model.py'' could be executed to evaluate test inputs and their neighborhoods and manipulations. The set of neighborhoods and manipulations would be defined in python class ``NeighMan'' (here is a set of noise on input image). The output of ``1-Eval_model.py'' would be results of inputs, neighborhoods and manipulations  for each classifier in a numpy file in ``Results'' folder.


In order to run ``1-Eval_model.py'', following inputs are required:
* Image dimension:  would be stored in ``img_width'', ``img_height'', ``img_num_channels'',
* The input images: would be stored in ``input_test'',
* Outputs of test inputs (and its neighborhoods and manipulations) for each classifier: would be stored numpy array in files ``Results'' folder,
* The number of neighborhoods and manipulations: would be stored in ``no_classes'',
* Ordered correct labels of inputs: would be stored in ``target_test''.
* The ``project_path'', ``data_dir'',  should be set to be the project path and test file path.


Then, using the output results in the ``Results'' folder, ``2-MASKS.py'' could be executed. Here, the correct output labels would be provided with the function ``load_labels''. After execution of this python code, following results for one to the number of classifiers would be provided:

* ``agent_counter'': number of agents,
* ``correct_answer'': correct verified answers,
* ``wrong_answer'': wrong verified answer,
* ``conflict_answer'': unverified answers because of conflicting,
* ``correct_assist'': unverified answers because more than one output classes are provided, but the correct answer is in the provided output classes,
* ``wrong_assist'':  unverified answers because more than one output classes are provided, but the correct answer is not in the provided output classes.

For further investigation, result of these multi-agent systems, for every input would be stored in ``Agents'' folder.


In order to run ``2-MASKS.py'', following inputs are required:
* Outputs of test inputs (and its neighborhoods and manipulations) for each classifier: would be stored numpy array in files ``Results'' folder.
* The number of output classes:
* be stored in ``no_classes''
* The number of neighborhoods and manipulations: would be stored in ``nei_man_no''
* Ordered correct labels of inputs: would be stored in ``target_test''.

The Modified Fashion MNIST, MNIST, and Fruit-360 examples could be found in https://github.com/iuwa/MASKS (FashionMNIST-MASKS.zip, MNIST-MASKS.zip, and Fruit-360-MASKS.zip).

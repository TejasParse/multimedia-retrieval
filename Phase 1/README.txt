Task 1:
you dont need anything to run task 1. just make sure the path you provide for the target video is correct

Task 2:
Task 2 is in a jupyter notebook. As long as the provided paths are all correct, the folders required for tasks 2b and 2c will automatically be generated when you run the preprocessing steps given in the file followed by task 2a. However, note that this will take time.

Task 3:
Same as Task 1, You only need to provide video path input to run task 3. The frames and features will be stored in the folders “Frames” and “Features” respectively and histogram will be saved in the task3_save folder. The folders will be automatically created when the task is run.

Task 4:
Task 4 will run only after you have run task 2 since it requires some of the folders created in task 2. Once all the above tasks are run, you can simply run all the cells of the file. NOTE THAT TASK3 WILL ONLY RUN IF YOU HAVE NUMPY VERSION - "1.23.2" AND TORCH - "2.4.1" due to compatibility issues.


Task 5: 
You can run task 5 after completing task 4. It will take a video file path as input and ask which model you want to run. Accordingly, it will compare the given video with all videos to find similar videos.


Please Note that the outputs are shown in the report file itself.
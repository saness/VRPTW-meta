The folder this README.txt is in consists of all the files needed to run Ant Colony Optimization algorithm to solve
Vehicle Routing Problem with Timed Constraint. The following are the steps to run the program:

main.py file: This is the main file that runs the program. Just run that file and the Ant Colony Optimization will start
            working on the problem. The algorithm runs for 30 trials and in each trail it runs for certain number of iterations.
            If it doesn't find any improvement for 300 iterations then that specific trial ends and another one begins.
            To change the number of trials just change the parameter "num_trials" to any number you want. In the end
            the program outputs the best distance and best number for vehicles from the 30 trails. The program also
            produces graphs of all best path and path distance found. It is saved in image folder under assets folder.

creategif.py: This file creates a gif out of all the graphs created in a trial. The gif is saved as output.gif in the same
                branch as other files. To create the gif just run the file.
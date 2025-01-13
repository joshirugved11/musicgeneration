# Music and Sound Generation
This repository is all about generating AI sound or music using the HuggingFace[**Transformers**] library and training the music generation model using **PyTorch**. 

The libraries used for this project are as specified 
1. `Transformers`[`HuggingFace`] - Using *pretrained models* for language and audio tasks
2. `PyTorch` - Deep Learning framework for *model building and training*
3. `Librosa` - *Audio analysis* and feature extraction
4. `Matplotlib` - Data visualisation for music analysis
5. `Pydub` - Audio *file conversion and manipulation*
6. `NumPy` - *Numerical operations* and matrix manipulation

# To use this project on windows, you need to do the following things 

1. When you will download the forked clone of this project you will need to install all the libraries in virtual environment, for this open your cloned project folder, click on the path of the folder and clear the path and type *cmd* the *command terminal* will open.

   To start virtual environment here are the commands
   `python -m venv .env`
   `.anv\Scripts\activate`
   after running project
   `deactivate`
   
1. **Fork** this repository on your local machine
   Now the repository link will be `https://github.com/<YourUserName>/projectname`

2. This will create a copy of the project on your local machine. Now that you have cloned the repo we will need to do two things: First is to make the necessary changes/contribution and commit those changes. After making your changes and adding new files, its time to add those changes into a separate branch before pushing them to remote. First let's create a branch. In your git bash, change the path to point to your repository directory. To do that use this command:

   `cd project folder name`
   `git checkout -b your-new-branch-name`
   `git checkout -b lary-mak`
   `git status`
   `git add`
   `git commit -m "<message here>"`

3. Push changes to remote Now that everything is set, it's time to let our maintainer know what we have added. That is made possible by pushing the changes with this command:

   `git push origin <add-your-branch-name>`

4. Submit changes If you go to your repository on GitHub and refresh the page, you'll see a Compare and pull request button. Click on that button.

# DETECTOR
1. make directory for your project (like example: mkdir henkel)
2. put to directory images and labels. example:
	
	```
	<project dir>/dataset/JPEGImages/*.jpg

	<project dir>/dataset/labels/*.txt
	```
3. put to project dir a config file. Name the config must be like present below:	

   ```custom.cfg```

   If you does not put config file to dir, script will using custom.cfg for training from tools directory.

4. run training command:
	
	```./train-detector.sh <project dir> <number of classes>```

# CLASSIFIER
1. make directory for your project (like example: mkdir shell)
2. put to directory images and labels. example:

        <project dir>/images/*.jpg
        
        <project dir>/names.txt

    Format file names.txt:
    
        person
        car
        dog

    Note! filename images must have name of class from file names.txt, like a example:
	
        000-person.jpg
        001-car.jpg
        002-dog.jpg
        003-person.jpg

3. put to project dir a config file. Name the config must be like present below:
    
    ```custom.cfg```

   Note! file custom.cfg must have structure for according number of classes

4. run training command:
    
    ```./train-classifier.sh <project dir> <number of classes>``` 

    exapmle: 
	
	```./train-classifier.sh visitors 184```


DESCRIPTION:
A program to convert scanned migraine diaries into data, using the template found in the PDF document in Template, let's call it DIARYTEMPLATE

OUTPUT:
A command line program written in python called ReadMigraineDiary

ReadMigraineDiary takes as argument a path to an image file containing the scanned headache diary and outputs a CSV file with the same name as the image file, but csv file suffix.
ReadMigraineDiary rejects the image if it doesn't fit a diary template to a reasonable certainty

the csv file has format:
column 1: date (assume it's the current year)
column 2: migraine (yes/no)
column 3: headache (yes/no)
column 4: medication (yes/no)

PROCESS:
To build this program you will do the following:
-  COLLECT TRAINING DATA
DIARYTEMPLATE contains 12 pages with dates on. Use these 12 pages to generate training data in a Training folder. For each of the 12 pages, generate images with ticks, crosses, and other realistic filling in of the headache diary with a pen/pencil. Generate different realistic poses, image conditions, resolutions to simulate what the average user would do if they were taking a picture of the diary on their phone. Store the ground truth for each image in the csv format above. Generate 30 different images for each of the 12 pages.
- Build ReadMigraineDiary using robust, non-hallucinating methods (e.g. RANSAC etc) to take an image and generate the CSV. 

ACCEPTANCE:
Keep improving ReadMigraineDiary until you can successfully match over 95% of the entries in the ground truth CSV for all the images.


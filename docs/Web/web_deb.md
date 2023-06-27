## Web Application Development:

To create a web application, we use the Django framework.

**The steps of this application are as follows:**

1. The user gets to the main page with a greeting and a warning;
2. Go to a page with an interface for recording a person from a front-facing webcam;
3. At the time of recording, the frames are sent to the server where our emotion recognition model works;
4. After we get the result and write it down;
5. When the recording stops and the user wants to see the result, he goes to the next page;
6. Output of the result on a graph with notation.

The application is packaged in an image, in order for the application to work, it is necessary to launch a container with TensorFlow Serving on which the machine learning model runs.

After the launch, the web-app will be available at - [http:localhost:8000](http:localhost:8000).
var isStream = false
var isRecord = false

var cameraInterval = null
var timerInterval = null
var listenerInterval = null

var result = null
var startTime = null
var time = null
// Set the params for the webcam
var constraints = { video: { width: 850, height: 500, facingMode: 'user' } }
// Get a video stream if the page is intended for recording
if (window.location.pathname === '/record/') {
    console.log("Webcam is working")
    navigator.mediaDevices.getUserMedia(constraints)
    .then(function(mediaStream) {
        // Set a video stream in the video tag
        var video = document.querySelector('video')
        video.srcObject = mediaStream
        video.onloadedmetadata = function(e) {
            video.play()
        }
        // Track the page
        isStream = true
        listenerInterval = setInterval(function() { listenPage() }, 1000)
    })
    .catch(function(err) {
        // Check for errors at the end.
        console.log(err.name + ': ' + err.message)
    })
} else if (window.location.pathname === '/result/') {
    // Get the result if the page changes
    getResult()
}

// The stream is closed if the user goes to another page
function listenPage() {
    if (isStream) {
        if (window.location.pathname !== '/record/') {
            isStream = false
            clearInterval(listenerInterval)
            console.log("Webcam was stopped")
        }
    }
}

// Start record stream
function recordStream() {
    console.log('Start record')
    var start_button = document.querySelector('#start_button')
    var stop_button = document.querySelector('#stop_button')
    var result_button = document.querySelector('#result_button')

    isRecord = true
    start_button.classList.add('d-none')
    stop_button.classList.remove('d-none')
    result_button.classList.add('d-none')
    // Init the result dict
    result = {angry: 0, disgust: 0, fear: 0, happy: 0, neutral: 0, sad: 0, surprise: 0, nothing: 0}
    // Send frames every second
    cameraInterval = setInterval(function() { sendFrame() }, 1000)
    // Start timer
    startTime = new Date().getTime()
    timerInterval = setInterval(function() { startTimer() }, 1000)
}

// Stop record
function stopRecordStream() {
    console.log('Stop record')
    var start_button = document.querySelector('#start_button')
    var stop_button = document.querySelector('#stop_button')
    var result_button = document.querySelector('#result_button')

    isRecord = false
    stop_button.classList.add('d-none')
    start_button.classList.remove('d-none')
    result_button.classList.remove('d-none')
    // Clear intervals
    clearInterval(cameraInterval)
    clearInterval(timerInterval)
    document.getElementById("timer").innerHTML = "0h 0m 0s"
    // Add the result and the recording time to the localStorage
    localStorage.setItem("result", JSON.stringify(result))
    localStorage.setItem("time", time)
}

// Send the POST request
async function postPredict(data) {
    try {
        const response = await fetch("/predict/", {
            method: "POST",
            body: data,
        })
        // Convert the response to a str and return
        const result = await response.text()
        return result
    } catch (error) {
        console.error("Error:", error)
        return null
    }
}

// Send frames to the server
async function sendFrame() {
    var canvas = document.querySelector('canvas')
    var video = document.querySelector('video')
    // Save the frame on the canvas
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)
    var dataURL = canvas.toDataURL()
    // Get bytes
    dataURL = dataURL.replace('data:image/png;base64,', '')
    // Send the POST request
    let response = await postPredict(dataURL)
    // Process the response, update the result
    if (response === 'None') {
        result['nothing'] = result['nothing'] + 1
//        localStorage.setItem("result", JSON.stringify(result))
    } else {
        result[response] = result[response] + 1
//        localStorage.setItem("result", JSON.stringify(result))
    }
}

// Update the countdown every second
function startTimer() {
    // Get today's date and time
    var currentTime = new Date().getTime()
    // Find the distance between now and the countdown date
    var distance = currentTime - startTime
    // Time calculations for hours, minutes and seconds
    var hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60))
    var minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60))
    var seconds = Math.floor((distance % (1000 * 60)) / 1000)
    // Display the result in the element with id="timer"
    time = hours + "h " + minutes + "m " + seconds + "s"
    document.getElementById("timer").innerHTML = time
}

// Get the result and display it
function getResult() {
    // Monitor the loading of the window
    window.onload = function() {
        // Get data from localStorage
        let rec_time = localStorage.getItem("time")
        let rec_res = localStorage.getItem("result")
        rec_res = JSON.parse(rec_res)
        if (rec_time) {
            // Display the time
            document.getElementById("time").innerHTML = rec_time
            // Draw the chart
            google.load("visualization", "1", {packages:["corechart"]})
            google.setOnLoadCallback(drawChart)
            function drawChart() {
                var data = google.visualization.arrayToDataTable([
                    ['Emotion', 'Graph of mood changes'],
                    ['Angry', rec_res['angry']],
                    ['Disgust', rec_res['disgust']],
                    ['Fear', rec_res['fear']],
                    ['Happy', rec_res['happy']],
                    ['Neutral', rec_res['neutral']],
                    ['Sad', rec_res['sad']],
                    ['Surprise', rec_res['surprise']],
                    ['Not recognized', rec_res['nothing']],
                ])
                var options = {
                    pieHole: 0.5,
                    backgroundColor: 'transparent',
                    colors: ['#a81e11','#316131','#3d1d59','#e07707','#848c8c','#2d3573','#874c80', '#1e1f1e',]
                }
                var chart = new google.visualization.PieChart(document.getElementById('donut_single'))
                chart.draw(data, options)
            }
        }
    }
}
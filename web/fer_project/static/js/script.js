var isStream = false
var isRecord = false
var cameraInterval = null
var timerInterval = null
var listenerInterval = null
var currentTime = null
const fps = 1
// Выбирает разрешение камеры
var constraints = { video: { width: 850, height: 500, facingMode: 'user' } }
var result = null
var time = null

if (window.location.pathname === '/record/') {
    console.log("Webcam is working")
    navigator.mediaDevices.getUserMedia(constraints)
    .then(function(mediaStream) {
        var video = document.querySelector('video')
        video.srcObject = mediaStream
        video.onloadedmetadata = function(e) {
            video.play()
        }
        isStream = true
        listenerInterval = setInterval(function() { listenPage() }, 1000)
    })
    .catch(function(err) {
        // check for errors at the end.
        console.log(err.name + ': ' + err.message)
    })
} else if (window.location.pathname === '/result/') {
    getResult()
}


function listenPage() {
    if (isStream) {
        if (window.location.pathname !== '/record/') {
            isStream = false
            clearInterval(listenerInterval)
            console.log("Webcam was stopped")
        }
    }
}


function recordStream() {
    console.log('Start record')
    var start_button = document.querySelector('#start_button')
    var stop_button = document.querySelector('#stop_button')
    var result_button = document.querySelector('#result_button')

    isRecord = true
    start_button.classList.add('d-none')
    stop_button.classList.remove('d-none')
    result_button.classList.add('d-none')

    result = {angry: 0, disgust: 0, fear: 0, happy: 0, neutral: 0, sad: 0, surprise: 0}

    cameraInterval = setInterval(function() { sendFrame() }, 1000/fps)
    currentTime = new Date().getTime()
    timerInterval = setInterval(function() { startTimer() }, 1000)
}


function stopRecordStream() {
    console.log('Stop record')
    var start_button = document.querySelector('#start_button')
    var stop_button = document.querySelector('#stop_button')
    var result_button = document.querySelector('#result_button')

    isRecord = false
    stop_button.classList.add('d-none')
    start_button.classList.remove('d-none')
    result_button.classList.remove('d-none')

    clearInterval(cameraInterval)
    clearInterval(timerInterval)
    document.getElementById("timer").innerHTML = "0h 0m 0s"

    localStorage.setItem("result", JSON.stringify(result))
    localStorage.setItem("time", time)
}


async function postPredict(data) {
    try {
        const response = await fetch("/predict/", {
            method: "POST",
            body: data,
        })

        const result = await response.text()
        return result
    } catch (error) {
        console.error("Error:", error)
        return null
    }
}


async function sendFrame() {
    var canvas = document.querySelector('canvas')
    var video = document.querySelector('video')
    var screenshotImage = document.querySelector('img')

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)
    var dataURL = canvas.toDataURL()
    screenshotImage.src = dataURL

    dataURL = dataURL.replace('data:image/png;base64,', '')

    let response = await postPredict(dataURL)
    if (response !== 'None') {
        result[response] = result[response] + 1
    }
}

// Update the count down every 1 second
function startTimer() {
    // Get today's date and time
    var now = new Date().getTime()

    // Find the distance between now and the count down date
    var distance = now - currentTime

    // Time calculations for days, hours, minutes and seconds
    var hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60))
    var minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60))
    var seconds = Math.floor((distance % (1000 * 60)) / 1000)

    // Display the result in the element with id="demo"
    time = hours + "h " + minutes + "m " + seconds + "s"
    document.getElementById("timer").innerHTML = time
}

function getResult() {
    window.onload = function() {
        let rec_time = localStorage.getItem("time")
        let rec_res = localStorage.getItem("result")
        rec_res = JSON.parse(rec_res)
        if (rec_time) {
            document.getElementById("time").innerHTML = rec_time

            google.load("visualization", "1", {packages:["corechart"]})
            google.setOnLoadCallback(drawChart)
            function drawChart() {
                var data = google.visualization.arrayToDataTable([
                    ['Emotion', 'Graph of mood changes'],
                    ['Angry', rec_res['angry']],
                    ['Disgust', rec_res['disgust']],
                    ['Fear', rec_res['fear']],
                    ['Happy', rec_res['happy']],
                    ['Neutral', rec_res['neural']],
                    ['Sad', rec_res['sad']],
                    ['Surprise', rec_res['surprise']],
                ])

                var options = {
                    pieHole: 0.5,
                    pieSliceTextStyle: {
                        color: 'black',
                    },
                    backgroundColor: 'transparent',
                    colors: ['#a81e11','#507350','#2c1970','#c9bd38','#89c5d6','#272e30','#b35415',]
                }

                var chart = new google.visualization.PieChart(document.getElementById('donut_single'))
                chart.draw(data, options)
            }
        }
    }
}
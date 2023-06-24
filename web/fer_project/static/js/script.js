var isStream = false
var isRecord = false
var cameraInterval = null
var timerInterval = null
var listenerInterval = null
var currentTime = null
const fps = 1
// Выбирает разрешение камеры
var constraints = { video: { width: 850, height: 500, facingMode: 'user' } }

if (window.location.pathname === '/test_webcam/') {
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
}


function listenPage() {
    if (isStream) {
        if (window.location.pathname !== '/test_webcam/') {
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

    isRecord = true
    start_button.classList.add('d-none')
    stop_button.classList.remove('d-none')
    console.log(isRecord)

    cameraInterval = setInterval(function() { sendFrame() }, 1000/fps)
    currentTime = new Date().getTime()
    timerInterval = setInterval(function() { startTimer() }, 1000)
}


function stopRecordStream() {
    console.log('Stop record')
    var start_button = document.querySelector('#start_button')
    var stop_button = document.querySelector('#stop_button')

    isRecord = false
    stop_button.classList.add('d-none')
    start_button.classList.remove('d-none')
    console.log(isRecord)

    clearInterval(cameraInterval)
    clearInterval(timerInterval)
    document.getElementById("timer").innerHTML = "0h 0m 0s"
}


async function postPredict(data) {
    try {
        const response = await fetch("/test_hello/", {
            method: "POST",
            body: data,
        })

        const result = await response.text()
        console.log("Success")
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

    let response = postPredict(dataURL)
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
    document.getElementById("timer").innerHTML = hours + "h " + minutes + "m " + seconds + "s"
}

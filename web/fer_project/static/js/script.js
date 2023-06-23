// Выбирает разрешение камеры
var constraints = { video: { width: 850, height: 500, facingMode: 'user' } }

navigator.mediaDevices.getUserMedia(constraints)
    .then(function(mediaStream) {
        var video = document.querySelector('video')
        video.srcObject = mediaStream
        video.onloadedmetadata = function(e) {
            video.play()
        }
    })
    .catch(function(err) {
        // check for errors at the end.
        console.log(err.name + ': ' + err.message)
    })

console.log(document.querySelector('button.start'))
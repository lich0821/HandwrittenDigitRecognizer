var canvas = document.getElementById("writting_panel");
var ctx = canvas.getContext("2d");
ctx.strokeStyle = "Black";
ctx.lineWidth = 15;

var startX,
    startY,
    x,
    y,
    borderWidth = 10,
    isDrawing = false;

$("#writting_panel")
    .mousedown(function(e) {
        isDrawing = true;
        startX = e.pageX - $(this).offset().left - borderWidth;
        startY = e.pageY - $(this).offset().top - borderWidth;
    })
    .mousemove(function(e) {
        if (!isDrawing) return;
        x = e.pageX - $(this).offset().left - borderWidth;
        y = e.pageY - $(this).offset().top - borderWidth;
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(x, y);
        ctx.stroke();
        startX = x;
        startY = y;
    })
    .mouseup(function() {
        isDrawing = false;
    })
    .mouseleave(function() {
        isDrawing = false;
    });

$("#erase").click(function() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});

$("#predict").click(function() {
    var base64data = canvas.toDataURL("image/jpeg;base64;").split(",")[1];
    var data = atob(base64data);
    var buff = new ArrayBuffer(data.length);
    var arr = new Uint8Array(buff);
    var blob, i, dataLen;

    for (i = 0, dataLen = data.length; i < dataLen; i++) {
        arr[i] = data.charCodeAt(i);
    }
    blob = new Blob([arr], { type: "image/jpeg" });

    formdata = new FormData();
    formdata.append("image", blob);

    $.ajax({
        type: "POST",
        url: "/upload",
        dataType: "json",
        processData: false,
        contentType: false,
        data: formdata,
        success: function(result) {
            console.dir(result);
            if (!result.saved) return;

            var img = $("<img>").attr({
                width: 28,
                height: 28,
                src: canvas.toDataURL()
            });

            $("#gallery").append(img.addClass("thumbnail"));
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        },
        error: function() {
            console.dir("error");
        }
    });
});

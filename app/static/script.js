var customBoard = new DrawingBoard.Board("custom-board", {
    controls: false,
    webStorage: false,
    background: "#F8F8FF",
    color: "black",
    size: 15,
    fillTolerance: 150
});

$("#erase").click(function() {
    customBoard.resetBackground();
    customBoard.clearWebStorage();
});

$("#predict").click(function() {
    var base64data = customBoard.getImg().split(",")[1];
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
        url: "/predict",
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
                src: customBoard.getImg()
            });

            $("#gallery").empty();
            $("#result").empty();
            $("#gallery").append(img.addClass("thumbnail"));
            $("#result").append(result.predict);

            customBoard.resetBackground();
            customBoard.clearWebStorage();
        },
        error: function() {
            console.dir("error");
        }
    });
});

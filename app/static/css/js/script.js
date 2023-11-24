function previewImage() {
    var fileInput = document.getElementById('fileInput');
    var imagePreview = document.getElementById('imagePreview');

    fileInput.addEventListener('change', function () {
        var file = fileInput.files[0];
        var reader = new FileReader();

        reader.onload = function (e) {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
        }

        reader.readAsDataURL(file);
    });
}

// Call the function to set up the image preview when the page loads
window.onload = previewImage;
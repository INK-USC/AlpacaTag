function activeon() {
  // Get the checkbox
  var checkBox = document.getElementById("onlinelearning");
  if (checkBox.checked == true){
    $('.active').attr('disabled',!(checkBox));
  } else {
    $('.active').attr('disabled',checkBox);
  }
}
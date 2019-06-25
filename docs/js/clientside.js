//import {Spinner} from 'spin.js';
var events = {}, eventAnnotations, event_keys;
var defaultEvent = "PersonX puts PersonX's trust in PersonY";
var beautifyRelations = {
  Xintent: "PersonX's intent",
  Xemotion: "PersonX's reaction",
  Otheremotion: "Other people's reaction"
};
$(".loadersmall").hide();

jQuery.get( "https://homes.cs.washington.edu/~msap/debug/event2mind/docs/data/event_noAnnots.php", function( data ) {
  var arr;
  $.each(data.split("\n"),function(i,line){
    arr = line.split(",",2);
    events[arr[1]] = arr[0];
  });
  event_keys = Object.keys(events).sort();
  updateSelector(true);
});


function updateSelector(full){
  var dataToAdd = full ? event_keys: event_keys.slice(event_keys.indexOf(defaultEvent), event_keys.indexOf(defaultEvent)+5);
  var dataToAppend = [];
  
  //console.log("Starting now!");
  $(".loadersmall").show();
  $.each(dataToAdd,function(i,k){
    var v = events[k];
    //$("#eventSelector").append('<option class="'+v+'"value="'+k+'" data-toggle="'+k+'">'+k+'</option>');
    dataToAppend.push('<option class="'+v+'"value="'+k+'" data-toggle="'+k+'">'+k+'</option>');
  });
  $("#eventSelector").append(dataToAppend);
  loadEvent(defaultEvent);
  //console.log("Done!");
  $("#eventSelector").prop("value",defaultEvent).selectpicker("refresh");
  //console.log("Done2!");
  $(".loadersmall").hide();
  //$("#eventSelecter").prop("value",defaultEventID).selectpicker("refresh");
}

function loadEvent(event) {
  $("#annotations").empty();
  $.get("https://homes.cs.washington.edu/~msap/debug/event2mind/docs/data/getEventAnnots.php?event="+event, function(data) {
    console.log(data);
    eventAnnotations = JSON.parse(data);
    $.each(eventAnnotations,function(k,d){
      $("#annotations").append("<p><strong>"+beautifyRelations[k]+"</strong>:&nbsp;"+d+"</p>");
    });
  });
}
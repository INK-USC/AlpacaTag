import axios from 'axios';
axios.defaults.xsrfCookieName = 'csrftoken';
axios.defaults.xsrfHeaderName = 'X-CSRFToken';
var baseUrl = window.location.href.split('/').slice(3, 5).join('/');
console.log(baseUrl);

var HTTP = axios.create({
  baseURL: `/api/${baseUrl}/`,
});

window.deleteWord = function(dictid){
    console.log(dictid);
    HTTP.delete(`history/${dictid}`).then((response) => {
        window.location.reload();
    });
}





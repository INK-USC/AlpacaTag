import Vue from 'vue';
import HTTP from './http';

const vm = new Vue({
  el: '#mail-app',
  delimiters: ['[[', ']]'],
  data: {
    embedding: 'glove',
    nounchunk: false,
    onlinelearning: false,
    history: false,
    active: '',
    batch: '',
    epoch: '',
    activeset: true,
  },
  methods: {
    created() {
    },
    save() {
      const payload = {
        embedding: this.embedding,
        nounchunk: this.nounchunk,
        onlinelearning: this.onlinelearning,
        history: this.history,
        active: this.active,
        batch: this.batch,
        epoch: this.epoch,
      };
      HTTP.post('settings/', payload).then((response) => {
        console.log(response);
      });
    },
    reset() {
      this.embedding ='glove';
      this.recommendation =[];
      this.active = '';
      this.batch = '';
      this.epoch = '';
      this.activeset= true;
    },
  },
  watch: {
    onlinelearning() {
      if (this.onlinelearning===false){
        this.activeset = true;
      }
      else{
        this.activeset = false;
        this.batch = '';
        this.epoch = '';
        this.active = '';
      }
    },
  },
});
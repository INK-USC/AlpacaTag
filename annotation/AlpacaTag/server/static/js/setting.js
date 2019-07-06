import Vue from 'vue';
import HTTP from './http';

const vm = new Vue({
  el: '#mail-app',
  delimiters: ['[[', ']]'],
  data: {
    embedding: 'glove',
    recommendation: [],
    active: '',
    batch: '',
    epoch: '',
    activeset: true,
  },
  methods: {
    created() {
    },
  },
  watch: {
    recommendation() {
      if (this.recommendation.includes("onlinelearning")){
        this.activeset = false;
      }
      else{
        this.activeset = true;
      }
    },
  },
});
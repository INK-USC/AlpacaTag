import Vue from 'vue';
import HTTP from './http';

const vm = new Vue({
  el: '#mail-app',
  delimiters: ['[[', ']]'],
  data: {
    embedding: 1,
    nounchunk: false,
    onlinelearning: false,
    history: false,
    active: 1,
    batch: null,
    epoch: null,
    acquire: null,
    tmpbatch: null,
    tmpepoch: null,
    tmpacquire : null,
    activeset: true,
  },
  methods: {
    save() {
      if (this.active === null) {
        this.active = 1;
      }
      else {
        this.active = this.active;
      }
      if (this.acquire === null) {
        this.tmpacquire = 5;
      }
      else {
        this.tmpacquire = this.acquire;
      }
      if (this.batch === null) {
        this.tmpbatch = 10;
      }
      else {
        this.tmpbatch = this.batch;
      }
      if (this.epoch === null) {
        this.tmpepoch = 5;
      }
      else {
        this.tmpepoch = this.epoch;
      }
      const payload = {
        embedding: this.embedding,
        nounchunk: this.nounchunk,
        onlinelearning: this.onlinelearning,
        history: this.history,
        active: this.active,
        batch: this.tmpbatch,
        epoch: this.tmpepoch,
        acquire: this.tmpacquire
      };
      HTTP.put(`settings/`, payload).then((response) => {
      });
    },
    reset() {
      this.embedding = 1;
      this.nounchunk = false;
      this.onlinelearning = false;
      this.history = false;
      this.active = 1;
      this.batch = null;
      this.epoch = null;
      this.acquire = null;
      this.activeset = true;
    },
  },
  watch: {
    onlinelearning() {
      if (this.onlinelearning === false) {
        this.activeset = true;
        this.batch = null;
        this.epoch = null;
        this.acquire = null;
        this.active = 1;
      } else {
        this.activeset = false;
      }
    },
    active() {
      if (this.active === 1) {
        this.acquire = null;
      }
    },
  },
  created() {
    HTTP.get('settings').then((response) => {
      console.log(response.data);
      this.embedding = response.data.embedding;
      this.nounchunk = response.data.nounchunk;
      this.onlinelearning = response.data.onlinelearning;
      this.history = response.data.history;
      this.active = response.data.active;
      this.batch = response.data.batch;
      this.epoch = response.data.epoch;
      this.acquire = response.data.acquire;
      if (this.onlinelearning === false) {
        this.batch = null;
        this.epoch = null;
        this.acquire = null;
      }
      if (this.active === 1) {
        this.acquire = null;
      }
    });
  },
});
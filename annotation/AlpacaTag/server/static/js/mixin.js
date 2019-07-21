import HTTP from './http';
import axios from 'axios';
axios.defaults.xsrfCookieName = 'csrftoken';
axios.defaults.xsrfHeaderName = 'X-CSRFToken';

const getOffsetFromUrl = function(url) {
  const offsetMatch = url.match(/[?#].*offset=(\d+)/);
  if (offsetMatch == null) {
    return 0;
  }
  return parseInt(offsetMatch[1], 10);
};

const storeOffsetInUrl = function(offset) {
  let href = window.location.href;

  const fragmentStart = href.indexOf('#') + 1;
  if (fragmentStart === 0) {
    href += '#offset=' + offset;
  } else {
    const prefix = href.substring(0, fragmentStart);
    const fragment = href.substring(fragmentStart);

    const newFragment = fragment.split('&').map(function(fragmentPart) {
      const keyValue = fragmentPart.split('=');
      return keyValue[0] === 'offset'
        ? 'offset=' + offset
        : fragmentPart;
    }).join('&');

    href = prefix + newFragment;
  }

  window.location.href = href;
};

const annotationMixin = {
  data() {
    return {
      pageNumber: 0,
      docs: [],
      annotations: [],
      recommendations: [],
      labels: [],
      onlineLearningIndices: new Set(),
      onlineLearningNum: 0,
      onlineLearningPer: 5,
      guideline: '',
      total: 0,
      remaining: 0,
      searchQuery: '',
      url: '',
      offset: getOffsetFromUrl(window.location.href),
      picked: 'all',
      count: 0,
      isActive: false,
      confirmtext: '',
      isLoading: true,
      loadingMsg: 'Loading...',
      embedding: 1,
      nounchunk: false,
      online: false,
      history: false,
      active: 1,
      batch: null,
      epoch: null,
      acquire: null,
      server: null,
      serverMsg: 'OFF',
      activeIndices: [],
      activeIndicesPerPage: [],
      activeScores: [],
      activeScoresPerPage: [],
    };
  },

  methods: {
    async nextPage() {
      this.pageNumber += 1;
      if (this.pageNumber === this.docs.length) {
        if (this.next) {
          this.url = this.next;
          if (this.onlineLearningIndices.size >= this.onlineLearningPer) {
            this.onlinelearning().then((res) => {
              this.onlineLearningIndices.clear();
              this.onlineLearningNum = 0;
              this.search();
            });
          }
          else {
            await this.search();
          }
          this.pageNumber = 0;
        } else if (this.active != 1) {
          this.activeIndicesPerPage.push(this.activeIndices);
          this.activeScoresPerPage.push(this.activeScores);
          await this.submit();
        } else {
          this.pageNumber = this.docs.length - 1;
        }
      }
    },

    async prevPage() {
      this.pageNumber -= 1;
      if (this.pageNumber === -1) {
        if (this.prev) {
          this.url = this.prev;
          if (this.onlineLearningIndices.size >= this.onlineLearningPer) {
            this.onlinelearning().then((res) => {
              this.onlineLearningIndices.clear();
              this.onlineLearningNum = 0;
              this.search();
            });
          }
          else{
            await this.search();
          }
          this.pageNumber = this.docs.length - 1;
        } else if (this.active != 1) {
          const state = this.getState();
          const last = this.activeIndicesPerPage.pop();
          const lastScore = this.activeScoresPerPage.pop();
          this.url = `docs/?q=${this.searchQuery}&is_checked=${state}&offset=${this.offset}&limit=${this.acquire}&active_indices=${last}`;
          await this.search();
          this.pageNumber = 0;
        } else {
          this.pageNumber = 0;
        }
      }
    },

    annotatedCheck(check) {
      const docId = this.docs[this.pageNumber].id;
      HTTP.patch(`docs/${docId}`, {'annotated': check}).then((response) => {
      });
    },

    confirm() {
      const check = this.docs[this.pageNumber].annotated;
      const docId = this.docs[this.pageNumber].id;
      if (!check) {
        this.annotatedCheck(true);
        this.docs[this.pageNumber].annotated = true;
        this.$refs["confirm"].style.backgroundColor = "#3cb371";
        this.confirmtext = "Confirmed";
        if (!this.onlineLearningIndices.has(docId)) {
          this.onlineLearningIndices.add(docId);
          this.onlineLearningNum = this.onlineLearningNum + 1;
        }
      }
      else {
        this.annotatedCheck(false);
        this.docs[this.pageNumber].annotated = false;
        this.$refs["confirm"].style.backgroundColor = "#cd5c5c";
        this.confirmtext = "Press this button if the sentence has no entities";
        HTTP.delete(`docs/${docId}/annotations/`).then((response) => {
          this.annotations[this.pageNumber].splice(0, this.annotations[this.pageNumber].length);
          if (this.onlineLearningIndices.has(docId)) {
            this.onlineLearningIndices.delete(docId);
            this.onlineLearningNum = this.onlineLearningNum - 1;
          }
        });
      }
      HTTP.get('progress').then((response) => {
        this.total = response.data.total;
        this.remaining = response.data.remaining;
      });
    },

    async process_data(response) {
      this.isLoading = true;
      this.loadingMsg="recommending";
      this.docs = response.data.results;
      this.next = response.data.next;
      this.prev = response.data.previous;
      this.count = response.data.count;
      this.annotations = [];
      this.recommendations = [];
      for (let i = 0; i < this.docs.length; i++) {
        const doc = this.docs[i];
        this.annotations.push(doc.annotations);
      }
      for (let i = 0; i < this.docs.length; i++) {
        await HTTP.get(`docs/${this.docs[i].id}/recommendations/`).then((recomm_response) => {
          const rec = recomm_response.data.recommendation;
          this.recommendations.push(rec);
        });
        this.isLoading = false;
      }
      this.offset = getOffsetFromUrl(this.url);
    },

    async search() {
      HTTP.get(this.url).then((response) => this.process_data(response));
    },

    getState() {
      if (this.picked === 'all') {
        return '';
      }
      if (this.picked === 'active') {
        return 'true';
      }
      return 'false';
    },

    async submit() {
      const state = this.getState();
      if (this.active != 1){
        this.activelearning().then((response) => {
            this.url = `docs/?q=${this.searchQuery}&is_checked=${state}&offset=${this.offset}&limit=${this.acquire}&active_indices=${this.activeIndices}`;
            this.search();
          });
      }
      else {
        this.url = `docs/?q=${this.searchQuery}&is_checked=${state}&offset=${this.offset}&limit=${this.acquire}`;
        await this.search();
      }
      this.pageNumber = 0;
    },

    async initiatelearning() {
      return await HTTP.get(`learninginitiate`).then((response) => {
        this.loadingMsg="initiate learning";
        console.log(response.data.isFirst);
      });
    },

    async activelearning() {
      return await HTTP.get(`activelearning`).then((response) => {
        this.loadingMsg="active learning";
        this.activeIndices = response.data.indices;
        for (let i = 0; i < this.activeIndices.length; i++) {
          this.activeIndices[i] = this.activeIndices[i] + 1;
        }
        this.activeScores = response.data.scores;
        console.log(this.activeIndices);
        console.log(this.activeScores);
      });
    },

    async connectServer(){
      return await HTTP.get('connectserver').then((response) => {
        this.server = response.data["connection"];
      });
    },

    removeLabel(annotation) {
      const docId = this.docs[this.pageNumber].id;
      HTTP.delete(`docs/${docId}/annotations/${annotation.id}`).then((response) => {
        const index = this.annotations[this.pageNumber].indexOf(annotation);
        this.annotations[this.pageNumber].splice(index, 1);
        if (this.annotations[this.pageNumber].length === 0) {
          this.docs[this.pageNumber].annotated = false;
          if (this.onlineLearningIndices.has(docId)) {
            this.onlineLearningIndices.delete(docId);
            this.onlineLearningNum = this.onlineLearningNum - 1;
          }
          HTTP.patch(`docs/${docId}`, {'annotated': false}).then((response) => {
          });
        }
      });
    },

    replaceNull(shortcut) {
      if (shortcut === null) {
        shortcut = '';
      }
      shortcut = shortcut.split(' ');
      return shortcut;
    },

    onlinelearning(){
      this.loadingMsg = "online learning";
      this.isLoading = true;
      const indices = Array.from(this.onlineLearningIndices);
      return HTTP.post(`onlinelearning/`, { 'indices': indices }).then((response) => {
        this.isLoading = false;
      });
    },
  },

  watch: {
    picked() {
      this.submit();
    },

    annotations() {
      const check = this.docs[this.pageNumber].annotated;
      if (!check) {
        this.$refs["confirm"].style.backgroundColor = "#cd5c5c";
        this.confirmtext = "Press this button if the sentence has no entities";
      }
      else {
        this.$refs["confirm"].style.backgroundColor = "#3cb371";
        this.confirmtext = "confirmed";
      }
      HTTP.get('progress').then((response) => {
        this.total = response.data.total;
        this.remaining = response.data.remaining;
      });
    },

    offset() {
      storeOffsetInUrl(this.offset);
    },
  },

  created() {
    HTTP.get('labels').then((response) => {
      this.labels = response.data;
    });
    HTTP.get('settings').then((response) => {
      this.embedding = response.data.embedding;
      this.nounchunk = response.data.nounchunk;
      this.online = response.data.onlinelearning;
      this.history = response.data.history;
      this.active = response.data.active;
      this.batch = response.data.batch;
      this.onlineLearningPer = response.data.batch;
      this.epoch = response.data.epoch;
      this.acquire = response.data.acquire;
      this.connectServer().then((response) => {
        if (this.online === true && this.server === true){
          this.initiatelearning().then((response) => {
            console.log("model server on");
            this.submit();
          });
        } else {
          console.log("model server off");
          this.submit();
        }
      });
    });
    HTTP.get().then((response) => {
      this.guideline = response.data.guideline;
    });
  },

  computed: {
    achievement() {
      const done = this.total - this.remaining;
      const percentage = Math.round(done / this.total * 100);
      return this.total > 0 ? percentage : 0;
    },

    compiledMarkdown() {
      return marked(this.guideline, {
        sanitize: true,
      });
    },

    id2label() {
      let id2label = {};
      for (let i = 0; i < this.labels.length; i++) {
        const label = this.labels[i];
        id2label[label.id] = label;
      }
      return id2label;
    },

    progressColor() {
      if (this.achievement < 30) {
        return 'is-danger';
      }
      if (this.achievement < 70) {
        return 'is-warning';
      }
      return 'is-primary';
    },

    achievementTrain() {
      const done = this.onlineLearningNum;
      const percentage = Math.round(done / this.onlineLearningPer * 100);
      return this.onlineLearningPer > 0 ? percentage : 0;
    },

    progressColorTrain() {
      if (this.achievementTrain < 30) {
        return 'is-danger';
      }
      if (this.achievementTrain < 70) {
        return 'is-warning';
      }
      return 'is-primary';
    },

    serverOn() {
      if (this.server === true) {
        this.serverMsg = 'ON';
        return 'is-primary';
      } else {
        this.serverMsg = 'OFF';
        return 'is-danger';
      }
    },
    nounOn() {
      if (this.nounchunk === true) {
        return 'is-primary';
      } else {
        return 'is-light';
      }
    },
    onlineOn() {
      if (this.online === true && this.server === true) {
        return 'is-primary';
      } else if (this.online === true && this.server === false) {
        return 'is-danger';
      } else {
        return 'is-light';
      }
    },
    historyOn() {
      if (this.history === true) {
        return 'is-primary';
      } else {
        return 'is-light';
      }
    },
    confidence() {
      if (this.active === 1) {
        return false;
      } else {
        return true;
      }
    },
    confidenceElement(){
      return this.activeScores[this.pageNumber];
    },
  },
};

export default annotationMixin;

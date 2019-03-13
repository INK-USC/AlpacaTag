const process = require('process');
const VueLoaderPlugin = require('vue-loader/lib/plugin')

module.exports = {
    mode: process.env.DEBUG === 'False' ? 'production' : 'development',
    entry: {
        'sequence_labeling': './static/js/sequence_labeling.js',
        'projects': './static/js/projects.js',
        'stats': './static/js/stats.js',
        'label': './static/js/label.js',
        'guideline': './static/js/guideline.js',
        'upload': './static/js/upload.js'
    },
    output: {
        path: __dirname + '/static/bundle',
        filename: '[name].js'
    },
    module: {
        rules: [
            {
                test: /\.vue$/,
                loader: 'vue-loader'
            }
        ]
    },
    plugins: [
        new VueLoaderPlugin()
    ],
    resolve: {
        extensions: ['.js', '.vue'],
        alias: {
            vue$: 'vue/dist/vue.esm.js',
        },
    },
}
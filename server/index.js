const express = require('express');
const cors = require('cors');
const {spawn} = require('child_process');
const path = require('path');
const mongoose = require('mongoose');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;

// 1. Middlewares
app.use(cors());
app.use(express.json());

// 2. In-Memory History Fallback (if MongoDB is offline)
let localHistory = [];

// 3. MongoDB Schema & Connection Setup
const predictionSchema = new mongoose.Schema({
    ticker: {type: String, required: true},
    prediction: {type: String, required: true},
    confidence: {type: Number, required: true},
    probability_up: {type: Number, required: true},
    latest_close: {type: Number, required: true},
    latest_date: {type: String, required: true},
    indicators: {type: Object},
    timestamp: {type: Date, default: Date.now}
});

const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/stock_predictor';
mongoose.connect(MONGO_URI)
    .then(() => console.log('🍃 Connected to MongoDB successfully!'))
    .catch(() => console.log('ℹ️  MongoDB not detected locally. Operating with in-memory storage fallback.'));


// Route 1: Health Check

app.get ('/api/health', (req, res) => {
    res.json({ status: 'ok', message: 'MERN Stock Predictor Backend is running!' });
});

// Route 2: Live AI Prediction Endpoint

app.get('/api/predict/:ticker', (req, res) => {
    const ticker = req.params.ticker || '^NSEI';
    console.log(`\n--> [API] Recieved prediction request for ticker: ${ticker}`);

    // Define the project root directory
    const projectRoot = path.resolve(__dirname, '..');
    const pythonExecutable = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');
    const scriptPath = path.join(projectRoot, 'src', 'predict.py');

    // Spawn Python with cwd set to project root!
    const pythonProcess = spawn(pythonExecutable, [scriptPath, '--ticker', ticker], {cwd: projectRoot});


    let outputData = '';
    let errorData = '';

    // Capture standard output from Pyton
    pythonProcess.stdout.on('data', (data) => {
        outputData += data.toString();
    });

    //  Capture standard error (warnings / errors)
    pythonProcess.stderr.on('data', (data) => {
        errorData+= data.toString();
    });

    pythonProcess.on('close', async (code) => {
        if (code !== 0) {
            console.error(`❌ Python process exited with error code ${code}`);
            return res.status(500).json({
                status: 'error',
                message: 'Failed to run prediction model.',
                details: errorData
            });
        }

        try {
            // Find the JSON block inside outputData (in case TensorFlow printed warnings)
            const jsonStartIndex = outputData.indexOf('{');
            const jsonEndIndex = outputData.lastIndexOf('}') + 1;
            const cleanJsonString = outputData.substring(jsonStartIndex, jsonEndIndex);

            const result = JSON.parse(cleanJsonString);

            // Log to MongoDB or fallback in-memory array

            const historyItem = {
                ticker: result.ticker,
                prediction: result.prediction,
                confidence: result.confidence,
                probability_up: result.probability_up,
                latest_close: result.latest_close,
                latest_date: result.latest_date,
                indicators: result.indicators,
                timestamp: new Date()
            };

            localHistory.unshift(historyItem);
            if (mongoose.connection.readyState === 1) {
                await Prediction.create(historyItem);
            }

            console.log(`✅ [API] Forecast completed fpr ${ticker}: ${result.prediction}(${result.confidence}%)`);
            res.json(result);

        } catch(parseError) {
            console.error ('❌ Failed to parse Python JSON output:', parseError);
            res.status(500).json({
                status: 'error',
                message: 'Error parsing model output.',
                raw: outputData
            });
        }
    });
});

// Route 3: Get Prediction History

app.get('/api/history', async (req, res) => {
    try {
        if (mongoose.connection.readyState ===1) {
            const history = await Prediction.find().sort({ timestamp: -1}).limit(20);
            return res.json(history);
        }
        res.json(localHistory.slice(0, 20));
    } catch (err) {
        res.status(500).json({status: 'error', message: err.message });
    }
});

// 4. Start Server
app.listen(PORT, () =>{
    console.log(`\n======================================================`);
    console.log(`🚀 Express Backend Server running on http://localhost:${PORT}`);
    console.log(`👉 Test API: http://localhost:${PORT}/api/predict/^NSEI`);
    console.log(`======================================================\n`);
});
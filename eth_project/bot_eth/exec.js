/*
 * exec.js
 *
 * This script provides a helper for submitting isolated, leveraged MARKET
 * orders to the Binance Futures REST API.  It encapsulates the API
 * authentication and signature process required by Binance and exposes a
 * simple ``placeOrder`` function.  The script is written in plain Node.js
 * without external dependencies – it relies on the built‑in ``https`` and
 * ``crypto`` modules.  Before running this script you must populate a
 * ``.env`` file (see ``.env.example``) with your Binance API key and secret.
 *
 * Note: This code is for demonstration purposes only.  Exercise caution
 * when trading real funds.  Always validate order parameters and handle
 * errors appropriately in production.
 */

const https = require('https');
const crypto = require('crypto');
const querystring = require('querystring');
require('dotenv').config();

const API_KEY = process.env.BINANCE_API_KEY;
const API_SECRET = process.env.BINANCE_API_SECRET;
const BASE_URL = 'fapi.binance.com'; // Binance Futures base domain

/**
 * Create a signature for the given query parameters using HMAC SHA256.
 * @param {string} queryString
 * @returns {string}
 */
function sign(queryString) {
  return crypto
    .createHmac('sha256', API_SECRET)
    .update(queryString)
    .digest('hex');
}

/**
 * Perform a signed POST request to the Binance Futures API.
 * @param {string} path
 * @param {Object} params
 * @returns {Promise<Object>}
 */
function signedPost(path, params) {
  const timestamp = Date.now();
  const query = { ...params, timestamp };
  const queryString = querystring.stringify(query);
  const signature = sign(queryString);
  const fullPath = `${path}?${queryString}&signature=${signature}`;
  const options = {
    hostname: BASE_URL,
    port: 443,
    path: fullPath,
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-MBX-APIKEY': API_KEY,
    },
  };
  return new Promise((resolve, reject) => {
    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => {
        data += chunk;
      });
      res.on('end', () => {
        try {
          const json = JSON.parse(data);
          resolve(json);
        } catch (err) {
          reject(err);
        }
      });
    });
    req.on('error', (e) => {
      reject(e);
    });
    req.end();
  });
}

/**
 * Place a MARKET order on ETHUSDT perpetual futures.
 *
 * @param {'BUY'|'SELL'} side
 * @param {number} quantity Number of contracts to buy or sell (in ETH)
 * @param {Object} [options]
 * @returns {Promise<Object>} API response
 */
async function placeOrder(side, quantity, options = {}) {
  const params = {
    symbol: 'ETHUSDT',
    side: side,
    type: 'MARKET',
    quantity: quantity.toString(),
    newOrderRespType: 'RESULT',
    isolation: 'TRUE',
    leverage: options.leverage || 55,
    recvWindow: 5000,
  };
  return signedPost('/fapi/v1/order', params);
}

module.exports = {
  placeOrder,
};
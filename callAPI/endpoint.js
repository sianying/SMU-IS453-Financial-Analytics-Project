export const URL = "http://127.0.0.1:5100";
export const URL_po = "http://127.0.0.1:5200";


export async function set_ticker_data(URL, body) {
    try {
        const data = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': "*"
            },
            body: JSON.stringify(body)
        }
        const response = await fetch(`${URL}/set_ticker_data`, data)
        if (response) {
            const result = await response.json()
            return result;
        }
    } catch(e) {
        const error = {
            "code": 404,
            "data": e
        }
        return error;
    }
    
}

export async function get_return_series(URL) {
    try {
        
        const response = await fetch(`${URL}/get_return_series`)
        if (response) {
            const result = await response.json()
            return result;
        }
    } catch(e) {
        const error = {
            "code": 404,
            "data": e
        }
        return error;
    }
}

export async function get_price(URL) {
    try {
        
        const response = await fetch(`${URL}/get_price`)
        if (response) {
            const result = await response.json()
            return result;
        }
    } catch(e) {
        const error = {
            "code": 404,
            "data": e
        }
        return error;
    }
}

export async function get_ema(URL) {
    try {
        
        const response = await fetch(`${URL}/get_ema`)
        if (response) {
            const result = await response.json()
            return result;
        }
    } catch(e) {
        const error = {
            "code": 404,
            "data": e
        }
        return error;
    }
}

export async function get_volatility(URL) {
    try {
        
        const response = await fetch(`${URL}/get_volatility`)
        if (response) {
            const result = await response.json()
            return result;
        }
    } catch(e) {
        const error = {
            "code": 404,
            "data": e
        }
        return error;
    }
}

export async function get_macd(URL) {
    try {
        
        const response = await fetch(`${URL}/get_macd`)
        if (response) {
            const result = await response.json()
            return result;
        }
    } catch(e) {
        const error = {
            "code": 404,
            "data": e
        }
        return error;
    }
}

export async function get_bollinger(URL) {
    try {
        
        const response = await fetch(`${URL}/get_bollinger`)
        if (response) {
            const result = await response.json()
            return result;
        }
    } catch(e) {
        const error = {
            "code": 404,
            "data": e
        }
        return error;
    }
}


export async function get_po(URL_po) {
    try {
        
        const response = await fetch(`${URL_po}/get_po`)
        if (response) {
            const result = await response.json()
            return result;
        }
    } catch(e) {
        const error = {
            "code": 404,
            "data": e
        }
        return error;
    }
    
}

export async function set_ticker_data_po(URL_po, body) {
    try {
        const data = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': "*"
            },
            body: JSON.stringify(body)
        }
        const response = await fetch(`${URL_po}/set_ticker_data`, data)
        if (response) {
            const result = await response.json()
            return result;
        }
    } catch(e) {
        const error = {
            "code": 404,
            "data": e
        }
        return error;
    }
    
}



from flask import Flask, render_template_string, request, Response
from virtual import LOCATIONS, generate_map, nx, G

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Virtual Campus Guide - ADCET Ashta</title>
    <style>
        body {font-family: Arial; text-align: center; background-color: #eef3f8;}
        h1 {background: #1e3a8a; color: white; padding: 12px;}
        form {margin: 20px;}
        select, button {
            padding: 10px; margin: 5px; font-size: 16px; border-radius: 8px;
            border: 1px solid #ccc; background-color: white;
        }
        button {background-color: #1e3a8a; color: white; cursor: pointer;}
        iframe {
            border: none; width: 92%; height: 520px; border-radius: 10px;
            box-shadow: 0px 0px 12px rgba(0,0,0,0.2);
        }
        .route-display {
            background-color: #fff; padding: 10px; margin: 10px auto;
            width: 60%; border-radius: 10px; box-shadow: 0 0 5px rgba(0,0,0,0.2);
        }
    </style>
</head>
<body>
    <h1>🏫 ADCET Virtual Campus Navigator</h1>
    <form method="get" action="/route_map">
        <label for="start"><b>From:</b></label>
        <select name="start" id="start" required title="Select start location">
            {% for key, loc in locations.items() %}
                <option value="{{ key }}" {% if key == start %}selected{% endif %}>{{ loc.name }}</option>
            {% endfor %}
        </select>
        <label for="end"><b>To:</b></label>
        <select name="end" id="end" required title="Select destination location">
            {% for key, loc in locations.items() %}
                <option value="{{ key }}" {% if key == end %}selected{% endif %}>{{ loc.name }}</option>
            {% endfor %}
        </select>
        <button type="submit">Show Route</button>
    </form>

    {% if route_text %}
        <div class="route-display"><b>Selected Route:</b> {{ route_text }}</div>
    {% endif %}

    {% if start and end %}
        <iframe src="/map_view?start={{ start }}&end={{ end }}"></iframe>
    {% endif %}
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_PAGE, locations=LOCATIONS, start=None, end=None, route_text=None)

@app.route("/route_map")
def route_map():
    start = request.args.get("start")
    end = request.args.get("end")

    try:
        path = nx.shortest_path(G, start, end, weight="weight")
        route_text = " ➜ ".join([LOCATIONS[p]["name"] for p in path])
    except nx.NetworkXNoPath:
        route_text = "No route available between selected locations."

    return render_template_string(HTML_PAGE, locations=LOCATIONS, start=start, end=end, route_text=route_text)

@app.route("/map_view")
def map_view():
    start = request.args.get("start")
    end = request.args.get("end")
    map_html = generate_map(start, end)
    return Response(map_html, mimetype="text/html")

if __name__ == "__main__":
    app.run(debug=True)

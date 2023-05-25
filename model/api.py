from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from model_deployment import predict
from flask_cors import CORS
from gevent.pywsgi import WSGIServer


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Movie Genre Classification',
    description='Movie Genre Classification')

ns = api.namespace('predict', 
     description='predict')

# Definición argumentos o parámetros de la API
parser = api.parser()

#txt
parser.add_argument(
    'txt', 
    type=str, 
    required=True, 
    help='txt', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

# Definición de la clase para disponibilización
@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {predict(args['txt'])}, 200
    
    
if __name__ == '__main__':
    #from waitress import serve
    #serve(app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000))
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    #app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
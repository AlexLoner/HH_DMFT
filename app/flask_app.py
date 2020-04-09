from flask import render_template, Flask, redirect, request
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField
from wtforms.validators import DataRequired
import os

SECRET_KEY = os.urandom(32)


# --------------------------------------------------------------------------------------------------
def input_num_at(string):
    return ' '.join(''.join([i if i.isdigit() else ' ' for i in string]).split()[:1])

def input_hop(string):
    s = ' '.join(''.join([i if i.isdigit() or i == '.' else ' ' for i in string]).split()[:1])
def input_vec(string):
    return ' '.join(''.join([i if i.isdigit() else ' ' for i in string]).split()[:3])


# --------------------------------------------------------------------------------------------------
class Vectors(FlaskForm):
    form_name = 'vec'
    n = 3
    for i in range(n):
        setattr(FlaskForm, form_name + str(i), StringField())


class MyForm(FlaskForm):
    pass


# --------------------------------------------------------------------------------------------------
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = SECRET_KEY
data = {'vecs': [],
        'atoms': [],
        'atoms_pos': [],
        'trans': [],
        'hop': []}


# ------------------------------------------------------
@app.route('/')
def root():
    return render_template("start_page.html")


# ------------------------------------------------------
@app.route('/fill_form', methods=['GET', 'POST'])
def choose_form():
    if request.method == "POST":
        if request.form['button'] == '2':
            return redirect('/fill_form/atoms')
        elif request.form['button'] == '1':
            return redirect('/fill_form/vecs')
        elif request.form['button'] == '3':
            return redirect('/fill_form/tranlations_hop')
    print(data)
    return render_template('fill_form.html')


# ------------------------------------------------------
@app.route('/fill_form/vecs', methods=['GET', 'POST'])
def vecs():
    form_vec = Vectors()

    if request.method == "POST":
        for f in form_vec:
            if f.name == "csrf_token": continue
            data['vecs'].append(input_vec(f.data))
        return redirect('/fill_form')

    if data['vecs'] == []:
        vec = ['0 1 2', '3 4 5', '6 7 8']
    else:
        vec = data['vecs']

    return render_template('form_vecs.html', form_vec=form_vec, vec=vec)


# ------------------------------------------------------
@app.route('/fill_form/atoms', methods=['GET', 'POST'])
def atoms():

    if request.method == 'POST':
        data['atoms'] = [int(request.form.get("atoms"))]
        return redirect('/fill_form/atoms/positions')

    if data['atoms']:
        atoms = data['atoms']
    else:
        atoms = [1]

    return render_template('form_atoms.html', atoms=atoms[0], sub=0)  # sub = 0 to show submit button


# ------------------------------------------------------
@app.route('/fill_form/atoms/positions', methods=['GET', 'POST'])
def atoms_pos():
    for i in range(data['atoms'][0]):
        setattr(MyForm, 'pos' + str(i), StringField())
    form_pos = MyForm()
    if request.method == "POST":
        for i in form_pos:
            if i.name[:3] == 'pos':
                data['atoms_pos'].append(input_vec(i.data))
        return redirect('/fill_form')
    return render_template('form_atom_pos.html', form_pos=form_pos, atoms=data['atoms'][0], sub=1)


# ------------------------------------------------------
@app.route('/fill_form/tranlations_hop', methods=['GET', 'POST'])
def trans_hop():
    tmp = 0
    for i in range(data['atoms'][0]):
        setattr(MyForm, 'tr_vec' + str(tmp), StringField())
        setattr(MyForm, 'hopping' + str(tmp), StringField())
        tmp += 1
    form_th = MyForm()

    if request.method == 'POST':
        if form_th['click'] == '+':
            setattr(MyForm, 'tr_vec' + str(tmp), StringField())
            setattr(MyForm, 'hopping' + str(tmp), StringField())
            tmp += 1
            form_th = MyForm()
            return render_template('form_trans_hop.html', form_th=form_th)

        elif form_th['click'] == '-':
            tmp -= 1
            delattr(MyForm, 'tr_vec' + str(tmp))
            delattr(MyForm, 'hopping' + str(tmp))
            form_th = MyForm()
            return  render_template('form_trans_hop.html', form_th=form_th)
        elif form_th['click'] == 'submit':
            for f in form_th:
                if f.name[:7] == "tr_vec":
                    data['trans'].append(input_vec(f.data))
                elif f.name[:7] == 'hopping':
                    data['hop'] = [input_num_at(f.data)]


            return redirect('/fill_form')
    return render_template('form_trans_hop.html', form_th=form_th)

# ------------------------------------------------------
@app.route('/tb_form', methods=['GET', 'POST'])
def tb_form():
    print(data)
    print(request.method)
    vecs = []
    if request.method == "POST":
        for i in range(3):
            vecs.append(input_vec(request.form.get('vec' + str(i + 1))))
        data['vecs'] = vecs
    return render_template('tb_form.html', vec=data["vecs"])

app.run()
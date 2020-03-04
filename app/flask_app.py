from flask import render_template, Flask, redirect, request
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField
from wtforms.validators import DataRequired
import os

SECRET_KEY = os.urandom(32)


# --------------------------------------------------------------------------------------------------
def input_num_at(string):
    return ' '.join(''.join([i if i.isdigit() else ' ' for i in string]).split()[:1])


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
app.n_vecs = 3
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
    return render_template('fill_form.html')


# ------------------------------------------------------
@app.route('/fill_form/vecs', methods=['GET', 'POST'])
def vecs():
    form_vec = Vectors()
    if request.method == "POST":
        for f in form_vec:
            if f.name == "csrf_token": continue
            data['vecs'].append(input_vec(f.data))

        print(data['vecs'])
        return redirect('/fill_form')
    if data['vecs'] == []:
        vec = ['0 1 2', '3 4 5', '6 7 8']
    else:
        vec = data['vecs']
    return render_template('form_vecs.html', form_vec=form_vec, vec=vec)  # n=3 - cartecian coordinates


# ------------------------------------------------------
@app.route('/fill_form/atoms', methods=['GET', 'POST'])
def atoms():
    if request.method == 'POST':
        data['atoms'].append(int(request.form.get("atoms")))

        print("data: ", data)
        return redirect('/fill_form/atoms/positions')
    if data['atoms']:
        atoms = data['atoms']
    else:
        atoms = [1]
    return render_template('form_atoms.html', atoms=atoms[0])


# ------------------------------------------------------
@app.route('/fill_form/atoms/positions', methods=['GET', 'POST'])
def atoms_pos():
    form_pos = MyForm()
    for i in range(data['atoms'][0]):
        setattr(form_pos, f'pos{i}', StringField())

    return render_template('form_atom_pos.html', atoms=data['atoms'][0])


# ------------------------------------------------------
@app.route('/fill_form/tranlations_hop', methods=['GET', 'POST'])
def trans_hop():
    return render_template('form_trans_hop.html')

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


# -------------------------------------------------------
# @app.route('/a', methods=['GET', 'POST'])
# def a():
#     form_vec = Vectors()
#     form_atom = Atoms()
#     print(form_vec.n)  # , form_atom.n)
#     print(vars(form_vec)['_fields'])
#     print(vars(form_atom)['_fields'])
#     if request.method == "POST":
#         vecs = []
#         for f in form_vec:
#             vecs.append(input_vec(f.data))
#             if len(vecs) == form_vec.n:
#                 break
#         data['vecs'] = vecs
#         data['atoms'] = input_num_at(form_atom.atoms.data)
#         return redirect('/')
#
#     return render_template('test.html', form_vec=form_vec, form_atom=form)


# -------------------------------------------------------
# @app.route('/form/2')
# def trans():

#     if request.form['but'] == "+":
#             app.n_vecs += 1
#             Vectors.n = app.n_vecs
#             form.add(app.n_vecs-1)
#             return redirect('/a')
#     elif request.form['but'] == '-' and app.n_vecs > 3:
#         app.n_vecs -= 1
#         Vectors.n = app.n_vecs
# #             print(app.n_vecs)
#         form.remove(app.n_vecs)
#         return redirect('/a')


# -------------------------------------------------------
# -------------------------------------------------------
# @app.route('/test_form', methods=['GET', 'POST'])
# def test_form():
#     print(request.form)
#     if request.method == 'POST':
#         plus = request.form.get('add_line')
#         if plus:
#             app.n_form += int(plus)
#             return render_template('test_form.html', n=app.n_form)
#     return render_template('test_form.html', n=app.n_form)

# ------------------------------------------------------

# @app.route('/test', methods=['GET', 'POST'])
# def test():
#     print(request.form)
#     if len(data['vecs']) < 3:
#         vecs = []
#         if request.method == "POST":
#             for i in range(3):
#                 vecs.append(input_vec(request.form.get('vec' + str(i+1))))
#             data['vecs'] = vecs
#         return render_template('test.html', vec = [])
#     else:
#         return render_template('test.html', vec = data["vecs"])
app.run()
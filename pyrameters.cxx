/*  This file is part of PKDGRAV3 (http://www.pkdgrav.org/).
 *  Copyright (c) 2001-2023 Douglas Potter & Joachim Stadel
 *
 *  PKDGRAV3 is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  PKDGRAV3 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with PKDGRAV3.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "pyrameters.h"

bool pyrameters::has(const char *name) const {
    bool bSpecified = false;
    if (auto f = PyObject_GetAttrString(specified_, name)) {
        bSpecified = PyObject_IsTrue(f)>0;
        Py_DECREF(f);
    }
    return bSpecified;
}

void pyrameters::merge(PyObject *o1, PyObject *o2) {
    auto d1 = PyObject_GenericGetDict(o1, nullptr);
    auto d2 = PyObject_GenericGetDict(o2, nullptr);
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(d2, &pos, &key, &value)) {
        PyDict_SetItem(d1, key, value);
    }
}

void pyrameters::merge(const pyrameters &other) {
    merge(this->arguments_, other.arguments_);
    merge(this->specified_, other.specified_);
}

bool pyrameters::verify(PyObject *kwobj) {
    bool bSuccess = true;
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(kwobj, &pos, &key, &value)) {
        const char *keyString;
        if (PyUnicode_Check(key)) {
            PyObject *ascii = PyUnicode_AsASCIIString(key);
            keyString = PyBytes_AsString(ascii);
            Py_DECREF(ascii);
            if (keyString[0]=='_') continue; // skip things that start with underscore
        }
        // If this key is not a valid argument, then print an error,
        // unless it is a module, or a callable imported from another module.
        if (!PyObject_HasAttr(arguments_, key) && !PyModule_Check(value)) {
            if (PyCallable_Check(value)) {
                auto module = PyObject_GetAttrString(value, "__module__");
                if (module && PyUnicode_Check(module)) {
                    auto ascii = PyUnicode_AsASCIIString(module);
                    auto result = PyBytes_AsString(ascii);
                    Py_DECREF(ascii);
                    Py_DECREF(module);
                    if (result && strcmp(result, "__main__")!=0) continue;
                }
            }
            PyErr_Format(PyExc_AttributeError, "invalid parameter %A", key);
            PyErr_Print();
            bSuccess = false;
        }
    }
    return bSuccess;
}

bool pyrameters::update(PyObject *kwobj, bool bIgnoreUnknown) {
    bool bSuccess = true;
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(kwobj, &pos, &key, &value)) {
        const char *keyString;
        if (PyUnicode_Check(key)) {
            PyObject *ascii = PyUnicode_AsASCIIString(key);
            keyString = PyBytes_AsString(ascii);
            Py_DECREF(ascii);
            if (keyString[0]=='_') continue;
        }
        if (PyObject_HasAttr(arguments_, key)) {
            PyObject_SetAttr(arguments_, key, value);
            PyObject_SetAttr(specified_, key, Py_True);
        }
        else if (!bIgnoreUnknown) {
            PyErr_Format(PyExc_AttributeError, "invalid parameter %A", key);
            PyErr_Print();
            bSuccess = false;
        }
    }
    return bSuccess;
}

template<> PyObject *pyrameters::get<PyObject *>(const char *name) const {
    auto v = PyObject_GetAttrString(arguments_, name);
    if (!v) throw std::domain_error(name);
    if (PyCallable_Check(v)) {
        auto callback = v;
        auto call_args = PyTuple_New(0);
        v = PyObject_Call(callback, call_args, dynamic_);
        Py_DECREF(call_args);
        Py_DECREF(callback);
    }
    if (PyErr_Occurred()) PyErr_Print();
    return v;
}

template<> void pyrameters::set_dynamic(const char *name, double value) {
    auto o = PyFloat_FromDouble(value);
    if (o) {
        PyDict_SetItemString(dynamic_, name, o);
        Py_DECREF(o);
    }
    if (PyErr_Occurred()) {
        PyErr_Print();
        abort();
    }
}

template<> void pyrameters::set_dynamic(const char *name, std::int64_t value) {
    auto o = PyLong_FromSsize_t(value);
    if (o) {
        PyDict_SetItemString(dynamic_, name, o);
        Py_DECREF(o);
    }
    if (PyErr_Occurred()) {
        PyErr_Print();
        abort();
    }
}

template<> void pyrameters::set_dynamic(const char *name, std::uint64_t value) {
    auto o = PyLong_FromSize_t(value);
    if (o) {
        PyDict_SetItemString(dynamic_, name, o);
        Py_DECREF(o);
    }
    if (PyErr_Occurred()) {
        PyErr_Print();
        abort();
    }
}

template<> void pyrameters::set_dynamic(const char *name, float value) {
    set_dynamic < double>(name, value);
}

template<> void pyrameters::set_dynamic(const char *name, std::int32_t value) {
    set_dynamic < std::int64_t>(name, value);
}

template<> void pyrameters::set_dynamic(const char *name, std::uint32_t value) {
    set_dynamic < std::uint64_t>(name, value);
}

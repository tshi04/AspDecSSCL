'''
@author Tian Shi
Please contact tshi@vt.edu
'''


def _is_digit(w):
    for ch in w:
        if not(ch.isdigit() or ch == ','):
            return False
    return True


def fix_tokenization(input_):

    input_ = input_.replace(' ##', '')
    input_ = input_.replace(" ` ` ` ", " `` ` ")
    input_ = input_.replace(" ' ' ' ", " ' '' ")
    input_ = input_.replace("` ` ` ", "`` ` ")
    input_ = input_.replace(" ' ' '", " ' ''")
    input_ = input_.replace(" ` ` ", " `` ")
    input_ = input_.replace(" ' ' ", " '' ")
    input_ = input_.replace("` ` ", "`` ")
    input_ = input_.replace(" ' '", " ''")
    input_ = input_.replace(' - - - ', ' --- ')
    input_ = input_.replace(' - - ', ' -- ')
    input_ = input_.replace(" ' d ", " 'd ")
    input_ = input_.replace(" ' ll ", " 'll ")
    input_ = input_.replace(" ' m ", " 'm ")
    input_ = input_.replace(" ' re ", " 're ")
    input_ = input_.replace(" ' s ", " 's ")
    input_ = input_.replace(" ' ve ", " 've ")
    input_ = input_.replace(" n ' t ", " n't ")
    input_ = input_.replace(" - year - old ", "-year-old ")
    input_ = input_.split()

    has_left_quote = False
    has_left_single_quote = False
    output = []
    i = 0
    flag = 0
    while i < len(input_):
        tok = input_[i]
        if tok == '"':
            if has_left_quote:
                output.append("''")
            else:
                output.append("``")
            has_left_quote = not has_left_quote
            i += 1
        elif tok == "'" and input_.count("'") == 1 and i < len(input_)-1 and len(output) > 0:
            if "`" in input_:
                output.append(tok)
                i += 1
            elif output[-1][-1] == 's':
                output.append(tok)
                i += 1
            else:
                output[-1] += tok + input_[i+1]
                i += 2
        elif tok == "`":
            has_left_single_quote = not has_left_single_quote
            output.append(tok)
            i += 1
        elif tok == "'" and input_.count("'") > 1:
            if has_left_single_quote:
                output.append("'")
            else:
                output.append("`")
            has_left_single_quote = not has_left_single_quote
            i += 1
        elif tok == "," and \
                len(output) > 0 and \
                _is_digit(output[-1]) and \
                i < len(input_) - 1 and \
                _is_digit(input_[i + 1]):
            # $ 3 , 000 -> $ 3,000
            output[-1] += ',' + input_[i + 1]
            i += 2
        elif tok == "." and \
                len(output) > 0 and \
                output[-1][-1].isdigit() and \
                i < len(input_) - 1 and \
                input_[i + 1].isdigit():
            # 3 . 03 -> $ 3.03
            output[-1] += '.' + input_[i + 1]
            i += 2
        elif tok == ":" and \
                len(output) > 0 and \
                output[-1][-1].isdigit() and \
                i < len(input_) - 1 and \
                input_[i + 1].isdigit():
            # 3 : 03 -> $ 3:03
            output[-1] += ':' + input_[i + 1]
            i += 2
        elif tok == "." and i < len(input_)-3:
            # . . . -> ...
            k = 1
            while i+k < len(input_):
                if input_[i+k] == ".":
                    k += 1
                else:
                    break
            # u . s . -> u.s.
            l = 1
            while i+l < len(input_)-1:
                if input_[i+l+1] == "." and \
                        input_[i+l] != "." and \
                        len(input_[i+l]) == 1:
                    l += 2
                else:
                    break
            if k > 1 and l > 1:
                output.append(''.join(input_[i:i+k]))
                i += k
            elif k > 1 and l == 1:
                output.append(''.join(input_[i:i+k]))
                i += k
            elif k == 1 and l > 1:
                output[-1] += ''.join(input_[i:i+l])
                i += l
            else:
                output.append(tok)
                i += 1
        elif tok == "*" and i < len(input_)-1:
            # * * * -> ***
            k = 1
            while i+k < len(input_):
                if input_[i+k] == "*":
                    k += 1
                else:
                    break
            output.append(''.join(input_[i:i+k]))
            i += k
        else:
            output.append(tok)
            i += 1

    if not output[-1] in ['.', '?', '!']:
        output.append('.')

    output = ' '.join(output)
    # output = output.replace(' - ', '-')
    # output = output.replace(' / ', '/')

    return output

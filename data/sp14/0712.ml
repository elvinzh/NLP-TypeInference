
let rec digitsOfInt n =
  if ((n mod 2) = 0) && (n > 0)
  then
    let rec loop input =
      if input = 0 then [] else (loop (input / 10)) @ [input mod 10] in
    loop n
  else [];;

let digits n = digitsOfInt (abs n);;

let rec sumList xs =
  if xs = [] then 0 else (let h::t = xs in h + (sumList t));;

let rec additivePersistence n =
  if (n / 10) = 0 then n else additivePersistence (sumList digits n);;


(* fix

let rec digitsOfInt n =
  if ((n mod 2) = 0) && (n > 0)
  then
    let rec loop input =
      if input = 0 then [] else (loop (input / 10)) @ [input mod 10] in
    loop n
  else [];;

let digits n = digitsOfInt (abs n);;

let rec sumList xs =
  if xs = [] then 0 else (let h::t = xs in h + (sumList t));;

let rec additivePersistence n =
  if (n / 10) = 0 then n else additivePersistence (sumList (digits n));;

*)

(* changed spans
(16,50)-(16,68)
(16,59)-(16,65)
*)

(* type error slice
(13,43)-(13,58)
(13,47)-(13,58)
(13,48)-(13,55)
(16,50)-(16,68)
(16,51)-(16,58)
*)

(* all spans
(2,20)-(8,9)
(3,2)-(8,9)
(3,5)-(3,31)
(3,5)-(3,20)
(3,6)-(3,15)
(3,7)-(3,8)
(3,13)-(3,14)
(3,18)-(3,19)
(3,24)-(3,31)
(3,25)-(3,26)
(3,29)-(3,30)
(5,4)-(7,10)
(5,17)-(6,68)
(6,6)-(6,68)
(6,9)-(6,18)
(6,9)-(6,14)
(6,17)-(6,18)
(6,24)-(6,26)
(6,32)-(6,68)
(6,52)-(6,53)
(6,32)-(6,51)
(6,33)-(6,37)
(6,38)-(6,50)
(6,39)-(6,44)
(6,47)-(6,49)
(6,54)-(6,68)
(6,55)-(6,67)
(6,55)-(6,60)
(6,65)-(6,67)
(7,4)-(7,10)
(7,4)-(7,8)
(7,9)-(7,10)
(8,7)-(8,9)
(10,11)-(10,34)
(10,15)-(10,34)
(10,15)-(10,26)
(10,27)-(10,34)
(10,28)-(10,31)
(10,32)-(10,33)
(12,16)-(13,59)
(13,2)-(13,59)
(13,5)-(13,12)
(13,5)-(13,7)
(13,10)-(13,12)
(13,18)-(13,19)
(13,25)-(13,59)
(13,37)-(13,39)
(13,43)-(13,58)
(13,43)-(13,44)
(13,47)-(13,58)
(13,48)-(13,55)
(13,56)-(13,57)
(15,28)-(16,68)
(16,2)-(16,68)
(16,5)-(16,17)
(16,5)-(16,13)
(16,6)-(16,7)
(16,10)-(16,12)
(16,16)-(16,17)
(16,23)-(16,24)
(16,30)-(16,68)
(16,30)-(16,49)
(16,50)-(16,68)
(16,51)-(16,58)
(16,59)-(16,65)
(16,66)-(16,67)
*)
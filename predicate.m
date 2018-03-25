function flag = predicate(region)
% Exemple 1 : On decoupe une region si l'ecart entre le pixel max et le
% pixel min depasse 0.2
%flag=(max(max(region))-min(min(region))>0.2);

% Exemple 2 : On decoupe une region si l'?cart type des pixels depasse 0.2
 flag=(std2(region)>0.05);
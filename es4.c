void rotazioneDestra(char *s, int k)
{
    int dim = 0;
    while (s[dim] != '\0')
        dim++;

    for (int j = 0; j < k; j++)
    {
        char ultimo = s[dim - 1];
        for (int i = dim - 1; i > 0; i--)
        {
            s[i] = s[i - 1];
        }
        s[0] = ultimo;
    }
}
